#!/usr/bin/env python3
"""
Comprehensive MoE Model Evaluation Script (Distributed)

Evaluates a trained MoE model across multiple dimensions:
1. Model accuracy (via lm_eval - runs directly on torchtitan model)
2. Computational cost (FLOPs, MFU)
3. Inference performance (latency, throughput)
4. Routing efficiency (load balance, expert utilization)

Requirements:
- GPU(s) with enough memory to load the model (uses same parallelism as training)
- DCP checkpoint (sharded checkpoints are automatically handled)

Usage:
    # Quick evaluation (skip lm_eval) - run with torchrun for multi-GPU
    torchrun --nproc_per_node=4 scripts/eval/eval_moe_model.py \
        --checkpoint_dir ./outputs/checkpoint/step-1000 \
        --config_file ./config.toml \
        --skip_lm_eval

    # Full evaluation with lm_eval (requires lm-eval installed)
    torchrun --nproc_per_node=4 scripts/eval/eval_moe_model.py \
        --checkpoint_dir ./outputs/checkpoint/step-1000 \
        --config_file ./config.toml
"""

# =============================================================================
# Imports
# =============================================================================

import argparse
import dataclasses
import importlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.flop_counter import FlopCounterMode
from torchtitan.components.metrics import build_device_memory_monitor
from torchtitan.config import ConfigManager, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.models.attention import create_attention_mask, get_causal_mask_mod
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.protocols.train_spec import get_train_spec
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.utils import device_module, device_type


# =============================================================================
# Rank-Aware Logging
# =============================================================================


class RankLogger:
    """Wrapper for rank-aware logging - only logs on rank 0."""

    def __init__(self, rank: int = 0):
        self.rank = rank

    def _should_log(self) -> bool:
        return self.rank == 0

    def info(self, msg: str):
        if self._should_log():
            logger.info(msg)

    def warning(self, msg: str):
        if self._should_log():
            logger.warning(msg)

    def error(self, msg: str):
        # Errors are logged on all ranks for debugging
        logger.error(msg)

    def debug(self, msg: str):
        if self._should_log():
            logger.debug(msg)


# Global rank logger - initialized after distributed setup
_rank_logger: RankLogger | None = None


def get_logger() -> RankLogger:
    """Get the rank-aware logger."""
    global _rank_logger
    if _rank_logger is None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        _rank_logger = RankLogger(rank)
    return _rank_logger


def init_rank_logger(rank: int):
    """Initialize the rank logger with the given rank."""
    global _rank_logger
    _rank_logger = RankLogger(rank)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EvalResults:
    """Container for all evaluation results."""

    routing_stats: dict[str, Any] | None = None
    latency_ms: float | None = None
    throughput_tokens_per_sec: float | None = None
    memory_allocated_gb: float | None = None
    memory_reserved_gb: float | None = None
    total_flops: int | None = None
    tflops: float | None = None
    active_params_b: float | None = None
    num_flops_per_token: float | None = None
    gpu_peak_flops: float | None = None
    lm_eval_results: dict[str, Any] | None = None
    walltime_seconds: float | None = None
    config: dict[str, Any] | None = None
    model_args: dict[str, Any] | None = None
    checkpoint_dir: str | None = None
    eval_timestamp: str | None = None
    output_file: str | None = None

    def to_dict(self) -> dict:
        """Convert results to dictionary for JSON serialization."""
        return {
            "routing_stats": self.routing_stats,
            "inference_performance": {
                "latency_ms": self.latency_ms,
                "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
                "memory_allocated_gb": self.memory_allocated_gb,
                "memory_reserved_gb": self.memory_reserved_gb,
            },
            "computational_cost": {
                "total_flops": self.total_flops,
                "tflops": self.tflops,
                "active_params_billions": self.active_params_b,
                "num_flops_per_token": self.num_flops_per_token,
                "gpu_peak_flops": self.gpu_peak_flops,
            },
            "lm_eval_results": self.lm_eval_results,
            "walltime_seconds": self.walltime_seconds,
            "config": self.config,
            "model_args": self.model_args,
            "checkpoint_dir": self.checkpoint_dir,
            "eval_timestamp": self.eval_timestamp,
            "output_file": self.output_file,
        }


# =============================================================================
# Routing Metrics Module
# =============================================================================


class RoutingMetrics:
    """Module for calculating MoE routing efficiency metrics."""

    @staticmethod
    def gini_coefficient(loads: list[float]) -> float:
        """Calculate Gini coefficient (0 = perfect equality, 1 = perfect inequality)."""
        sorted_loads = sorted(loads)
        n = len(sorted_loads)
        cumsum = sum((idx + 1) * val for idx, val in enumerate(sorted_loads))
        total = sum(sorted_loads)
        return (2 * cumsum) / (n * total) - (n + 1) / n if total > 0 else 0.0

    @staticmethod
    def calculate_layer_metrics(tokens_per_expert) -> dict[str, Any]:
        """Calculate all routing metrics for a single MoE layer."""
        mean_load = float(tokens_per_expert.mean())
        std_load = float(tokens_per_expert.std())
        num_experts = len(tokens_per_expert)
        active_experts = int((tokens_per_expert > 0).sum())

        return {
            "total_tokens_routed": int(tokens_per_expert.sum()),
            "num_experts": num_experts,
            "mean_tokens_per_expert": mean_load,
            "std_tokens_per_expert": std_load,
            "max_tokens_per_expert": int(tokens_per_expert.max()),
            "min_tokens_per_expert": int(tokens_per_expert.min()),
            "coefficient_of_variation": std_load / mean_load if mean_load > 0 else 0.0,
            "expert_utilization_rate": active_experts / num_experts,
            "gini_coefficient": RoutingMetrics.gini_coefficient(tokens_per_expert.tolist()),
            "tokens_per_expert_distribution": tokens_per_expert.tolist(),
        }

    @staticmethod
    def aggregate_stats(layer_stats: dict[str, dict]) -> dict[str, Any]:
        """Calculate aggregate statistics across all MoE layers."""
        if not layer_stats:
            return {"num_moe_layers": 0}

        cvs = [s["coefficient_of_variation"] for s in layer_stats.values()]
        ginis = [s["gini_coefficient"] for s in layer_stats.values()]
        utils_rates = [s["expert_utilization_rate"] for s in layer_stats.values()]

        return {
            "num_moe_layers": len(layer_stats),
            "avg_coefficient_of_variation": sum(cvs) / len(cvs),
            "avg_gini_coefficient": sum(ginis) / len(ginis),
            "avg_expert_utilization_rate": sum(utils_rates) / len(utils_rates),
        }


# =============================================================================
# FLOPs Calculation Module
# =============================================================================


class FlopsCalculator:
    """Module for calculating computational cost (FLOPs)."""

    @staticmethod
    def from_model_args(model, model_args, seq_len: int) -> tuple[int | None, float | None]:
        """Get theoretical FLOPs from model_args if available."""
        if not hasattr(model_args, "get_nparams_and_flops"):
            return None, None

        try:
            _, num_flops_per_token = model_args.get_nparams_and_flops(model, seq_len)
            total_flops = num_flops_per_token * seq_len
            get_logger().info(f"Theoretical FLOPs per token: {num_flops_per_token:.2e}")
            return int(total_flops), num_flops_per_token
        except Exception as e:
            get_logger().warning(f"Could not get theoretical FLOPs: {e}")
            return None, None

    @staticmethod
    def measure_with_counter(model, input_ids, attention_mask) -> int | None:
        """Measure FLOPs using PyTorch's FlopCounterMode."""
        try:
            flop_counter = FlopCounterMode(display=False)
            with flop_counter:
                _ = model(input_ids, attention_masks=attention_mask)
            measured = flop_counter.get_total_flops()
            get_logger().info(f"Measured FLOPs: {measured:.2e}")
            return measured
        except Exception as e:
            get_logger().warning(f"FlopCounterMode failed: {type(e).__name__}")
            return None

    @staticmethod
    def estimate_from_params(total_params: int, seq_len: int) -> int:
        """Rough FLOP estimate based on parameter count."""
        total_flops = 2 * total_params * seq_len
        get_logger().warning(f"Using rough FLOP estimate: {total_flops / 1e12:.2f} TFLOPs")
        return total_flops


# =============================================================================
# Input Preparation
# =============================================================================


def prepare_input_ids(tokenizer, prompts: list[str] | None, vocab_size: int, device) -> torch.Tensor:
    """Prepare input token IDs from prompts or generate random tokens."""
    log = get_logger()

    if tokenizer is None:
        log.warning("No tokenizer. Using random token IDs for benchmark.")
        return torch.randint(0, vocab_size, (1, 64), device=device)

    if prompts is None:
        prompts = [
            "The future of artificial intelligence is",
            "In the field of quantum computing,",
            "Machine learning models have revolutionized",
        ]

    try:
        encoded = [tokenizer.encode(p) for p in prompts]
        max_len = max(len(e) for e in encoded)
        pad_id = getattr(tokenizer, "pad_id", 0)
        padded = [e + [pad_id] * (max_len - len(e)) for e in encoded]
        log.info(f"Tokenized {len(prompts)} prompts, max length: {max_len}")
        return torch.tensor(padded, device=device)
    except (TypeError, AttributeError) as e:
        log.warning(f"Tokenizer encode failed: {e}. Using random IDs.")
        return torch.randint(0, vocab_size, (1, 64), device=device)


# =============================================================================
# Results Formatting & Printing
# =============================================================================


class ResultsPrinter:
    """Module for printing evaluation results (rank 0 only)."""

    def __init__(self, rank: int):
        self.rank = rank

    def _print(self, *args, **kwargs):
        """Print only on rank 0."""
        if self.rank == 0:
            print(*args, **kwargs)

    @staticmethod
    def format_walltime(seconds: float) -> str:
        """Format walltime as HH:MM:SS.ss."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"

    def print_section(self, title: str, items: list[tuple[str, Any]]):
        """Print a section with title and key-value items."""
        self._print(f"\n[{title}]")
        self._print("-" * 80)
        for label, value in items:
            if value is not None:
                self._print(f"  {label:<35} {value}")

    def print_routing_summary(self, routing_stats: dict | None):
        """Print routing efficiency summary."""
        if not routing_stats:
            self.print_section("ROUTING EFFICIENCY", [("Status", "No data available")])
            return

        agg = routing_stats.get("aggregate", {})
        self.print_section("ROUTING EFFICIENCY", [
            ("Average Coefficient of Variation:", f"{agg.get('avg_coefficient_of_variation', 0):.4f}"),
            ("Average Gini Coefficient:", f"{agg.get('avg_gini_coefficient', 0):.4f}"),
            ("Average Expert Utilization:", f"{agg.get('avg_expert_utilization_rate', 0):.2%}"),
            ("Number of MoE Layers:", agg.get('num_moe_layers', 0)),
        ])

    def print_performance_summary(self, results: EvalResults):
        """Print inference performance summary."""
        if results.latency_ms is None:
            self.print_section("INFERENCE PERFORMANCE", [("Status", "No data available")])
            return

        self.print_section("INFERENCE PERFORMANCE", [
            ("Latency:", f"{results.latency_ms:.2f} ms"),
            ("Throughput:", f"{results.throughput_tokens_per_sec:.2f} tokens/s"),
            ("Memory Allocated:", f"{results.memory_allocated_gb:.2f} GB"),
            ("Memory Reserved:", f"{results.memory_reserved_gb:.2f} GB"),
        ])

    def print_cost_summary(self, results: EvalResults):
        """Print computational cost summary."""
        items = []
        if results.tflops is not None:
            items.append(("Total FLOPs:", f"{results.tflops:.2f} TFLOPs"))
        if results.active_params_b is not None:
            items.append(("Active Parameters:", f"{results.active_params_b:.2f}B"))
        if results.num_flops_per_token:
            items.append(("FLOPs per Token:", f"{results.num_flops_per_token:.2e}"))

        if items:
            self.print_section("COMPUTATIONAL COST", items)

    def print_lm_eval_summary(self, lm_eval_results: dict | None):
        """Print lm_eval results summary."""
        if not lm_eval_results:
            return

        self._print("\n[MODEL ACCURACY (lm_eval)]")
        self._print("-" * 80)
        for task, metrics in lm_eval_results.items():
            if isinstance(metrics, dict):
                self._print(f"  {task}:")
                for metric, value in metrics.items():
                    self._print(f"    {metric}: {value}")

    def print_full_summary(self, results: EvalResults):
        """Print complete evaluation summary."""
        self._print("\n" + "=" * 80)
        self._print("EVALUATION SUMMARY")
        self._print("=" * 80)

        self.print_routing_summary(results.routing_stats)
        self.print_performance_summary(results)
        self.print_cost_summary(results)
        self.print_lm_eval_summary(results.lm_eval_results)

        if results.walltime_seconds:
            self.print_section("WALLTIME", [
                ("Total Time:", f"{self.format_walltime(results.walltime_seconds)} ({results.walltime_seconds:.2f}s)")
            ])

        if results.output_file:
            self.print_section("OUTPUT", [("Results File:", results.output_file)])

        self._print("\n" + "=" * 80)


# =============================================================================
# Config Extraction
# =============================================================================


def extract_config_for_results(job_config) -> dict[str, Any]:
    """Extract key config sections for serialization."""
    return {
        "job": {
            "description": job_config.job.description,
            "dump_folder": job_config.job.dump_folder,
        },
        "model": {
            "name": job_config.model.name,
            "flavor": job_config.model.flavor,
        },
        "training": {
            "seq_len": job_config.training.seq_len,
            "local_batch_size": job_config.training.local_batch_size,
        },
        "parallelism": {
            "data_parallel_shard_degree": job_config.parallelism.data_parallel_shard_degree,
            "data_parallel_replicate_degree": job_config.parallelism.data_parallel_replicate_degree,
            "tensor_parallel_degree": job_config.parallelism.tensor_parallel_degree,
            "pipeline_parallel_degree": job_config.parallelism.pipeline_parallel_degree,
            "expert_parallel_degree": job_config.parallelism.expert_parallel_degree,
            "context_parallel_degree": job_config.parallelism.context_parallel_degree,
        },
    }


# =============================================================================
# MoE Evaluator Class
# =============================================================================


class MoEEvaluator:
    """Distributed Evaluator for MoE models."""

    def __init__(self, checkpoint_dir: str, config_file: str, seed: int = 1337):
        self.checkpoint_dir = Path(checkpoint_dir)
        self._init_logging()
        self._load_config(config_file)
        self._init_environment(seed)

        # Initialize rank-aware logger after distributed setup
        init_rank_logger(self.rank)
        self.log = get_logger()
        self.printer = ResultsPrinter(self.rank)

        self._init_model_args()
        self._init_device_monitor()
        self._init_tokenizer()
        self.model = self._load_model()

    # -------------------------------------------------------------------------
    # Initialization Methods
    # -------------------------------------------------------------------------

    def _init_logging(self):
        """Initialize logging and API usage tracking."""
        torch._C._log_api_usage_once("torchtitan.eval_moe_model")

    def _load_config(self, config_file: str):
        """Load configuration from TOML file."""
        config_manager = ConfigManager()
        self.job_config = config_manager.parse_args([f"--job.config_file={config_file}"])

        if self.job_config.experimental.custom_import:
            importlib.import_module(self.job_config.experimental.custom_import)

    def _init_environment(self, seed: int):
        """Initialize distributed or single-process environment."""
        self.is_distributed = "LOCAL_RANK" in os.environ

        if self.is_distributed:
            self._init_distributed_env()
        else:
            self._init_single_process_env(seed)

    def _init_distributed_env(self):
        """Initialize distributed environment (launched with torchrun)."""
        local_rank = int(os.environ["LOCAL_RANK"])
        self.device = torch.device(f"{device_type}:{local_rank}")
        device_module.set_device(self.device)

        self.parallel_dims = self._init_distributed()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self._validate_parallelism()

        dist_utils.set_determinism(
            self.parallel_dims,
            self.device,
            self.job_config.debug,
            distinct_seed_mesh_dims=["pp"],
        )

    def _init_single_process_env(self, seed: int):
        """Initialize single-process environment (CPU or single GPU)."""
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            device_module.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        self.rank = 0
        self.world_size = 1
        self.parallel_dims = None

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _init_distributed(self) -> ParallelDims:
        """Initialize distributed process group and return ParallelDims."""
        job_config = self.job_config
        world_size = dist_utils.init_distributed(
            job_config.comm,
            enable_cpu_backend=job_config.training.enable_cpu_offload,
            base_folder=job_config.job.dump_folder,
        )

        p = job_config.parallelism
        return ParallelDims(
            dp_shard=p.data_parallel_shard_degree,
            dp_replicate=p.data_parallel_replicate_degree,
            cp=p.context_parallel_degree,
            tp=p.tensor_parallel_degree,
            pp=p.pipeline_parallel_degree,
            ep=p.expert_parallel_degree,
            etp=p.expert_tensor_parallel_degree,
            world_size=world_size,
        )

    def _validate_parallelism(self):
        """Validate world_size matches parallelism requirements."""
        p = self.job_config.parallelism
        ep, tp, pp = p.expert_parallel_degree, p.tensor_parallel_degree, p.pipeline_parallel_degree
        min_required = ep * tp * pp

        if min_required > self.world_size:
            raise RuntimeError(
                f"Config requires {min_required} GPUs (EP={ep} x TP={tp} x PP={pp}), "
                f"but only {self.world_size} processes launched."
            )

    def _init_model_args(self):
        """Initialize model arguments from train spec."""
        model_name = self.job_config.model.name
        model_flavor = self.job_config.model.flavor

        self.train_spec = get_train_spec(model_name)
        self.model_args = self.train_spec.model_args[model_flavor]
        self.model_args.update_from_config(self.job_config)

        self.log.info(f"Building {model_name} {model_flavor}")

    def _init_device_monitor(self):
        """Initialize device memory monitor and GPU peak flops."""
        self.device_memory_monitor = build_device_memory_monitor()
        self.gpu_peak_flops = utils.get_peak_flops(self.device_memory_monitor.device_name)

    def _init_tokenizer(self):
        """Initialize tokenizer from train spec."""
        self.tokenizer = (
            self.train_spec.build_tokenizer_fn(self.job_config)
            if self.train_spec.build_tokenizer_fn is not None
            else None
        )

    # -------------------------------------------------------------------------
    # Model Loading
    # -------------------------------------------------------------------------

    def _load_model(self) -> torch.nn.Module:
        """Load model from checkpoint with parallelization."""
        self.log.info(f"Loading model from {self.checkpoint_dir}")

        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {self.checkpoint_dir}")

        model_dtype = TORCH_DTYPE_MAP[self.job_config.training.dtype]

        with torch.device("meta"), utils.set_default_dtype(model_dtype):
            model = self.train_spec.model_cls(self.model_args)

        if self.is_distributed:
            model = self._load_distributed_model(model)
        else:
            model = self._load_single_process_model(model)

        model.eval()
        total_params = sum(p.numel() for p in model.parameters())
        self.log.info(f"Total parameters (this rank): {total_params / 1e9:.2f}B")
        return model

    def _load_distributed_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Load model in distributed mode with parallelization."""
        model_converters = build_model_converters(self.job_config, self.parallel_dims)
        model_converters.convert(model)

        if self.parallel_dims.pp_enabled:
            raise NotImplementedError("Pipeline parallel evaluation is not yet supported.")

        if self.train_spec.parallelize_fn is not None:
            self.train_spec.parallelize_fn(model, self.parallel_dims, self.job_config)

        init_device, buffer_device = self._get_init_devices()
        model.to_empty(device=init_device)
        with torch.no_grad():
            model.init_weights(buffer_device=buffer_device)
        model.train()  # Required for proper DTensor initialization

        self._load_checkpoint_dcp(model)
        return model

    def _load_single_process_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Load model in single-process mode."""
        model.to_empty(device=self.device)
        with torch.no_grad():
            model.init_weights()

        self._load_checkpoint_dcp(model)
        return model

    def _get_init_devices(self) -> tuple[str, str | None]:
        """Get initialization device and buffer device."""
        if self.job_config.training.enable_cpu_offload:
            return "cpu", device_type
        return device_type, None

    def _load_checkpoint_dcp(self, model: torch.nn.Module):
        """Load checkpoint using DCP."""
        self.log.info("Loading checkpoint with DCP...")
        state_dict = model.state_dict()
        dcp.load(state_dict, checkpoint_id=str(self.checkpoint_dir))
        model.load_state_dict(state_dict, assign=True)

    # -------------------------------------------------------------------------
    # Attention Mask Creation
    # -------------------------------------------------------------------------

    def _create_attention_mask(self, input_ids: torch.Tensor):
        """Create attention mask for model forward pass."""
        attn_type = getattr(self.model_args, "attn_type", "sdpa")
        if attn_type not in ["flex", "varlen"]:
            return None

        if hasattr(self.model, "get_attention_masks") and self.tokenizer is not None:
            return self.model.get_attention_masks(input_ids, self.tokenizer)

        B, seq_len = input_ids.shape
        mask_mod = get_causal_mask_mod()
        return create_attention_mask(mask_mod, B, None, seq_len, seq_len)

    # -------------------------------------------------------------------------
    # Evaluation Methods
    # -------------------------------------------------------------------------

    def evaluate_routing_efficiency(self, num_samples: int = 100, seq_len: int = 512) -> dict[str, Any]:
        """Evaluate MoE routing efficiency and load balance."""
        self.log.info("Evaluating routing efficiency...")

        # Reset counters
        for layer in self.model.layers.values():
            if hasattr(layer, "moe") and layer.moe is not None:
                layer.moe.tokens_per_expert.zero_()

        # Run inference with progress bar
        self._run_with_progress(
            num_samples,
            lambda: self._routing_inference_step(seq_len),
            "Routing analysis",
        )

        # Collect stats
        layer_stats = {}
        for i, layer in enumerate(self.model.layers.values()):
            if hasattr(layer, "moe") and layer.moe is not None:
                tokens_per_expert = layer.moe.tokens_per_expert.cpu().numpy()
                layer_stats[f"layer_{i}"] = RoutingMetrics.calculate_layer_metrics(tokens_per_expert)

        layer_stats["aggregate"] = RoutingMetrics.aggregate_stats(
            {k: v for k, v in layer_stats.items() if k != "aggregate"}
        )

        agg = layer_stats["aggregate"]
        self.log.info(
            f"Routing: avg_cv={agg.get('avg_coefficient_of_variation', 0):.3f}, "
            f"avg_gini={agg.get('avg_gini_coefficient', 0):.3f}"
        )

        return layer_stats

    def _routing_inference_step(self, seq_len: int):
        """Single routing inference step."""
        input_ids = torch.randint(0, self.model_args.vocab_size, (1, seq_len), device=self.device)
        attention_mask = self._create_attention_mask(input_ids)
        with torch.no_grad():
            _ = self.model(input_ids, attention_masks=attention_mask)

    def evaluate_inference_performance(
        self,
        prompts: list[str] | None = None,
        max_new_tokens: int = 128,
        num_warmup: int = 3,
        num_iterations: int = 10,
    ) -> dict[str, float]:
        """Evaluate inference performance (latency, throughput, memory)."""
        self.log.info("Evaluating inference performance...")

        input_ids = prepare_input_ids(self.tokenizer, prompts, self.model_args.vocab_size, self.device)

        # Warmup
        self._run_with_progress(num_warmup, lambda: self._generate(input_ids, 32), "Warmup")

        # Benchmark
        if self.device.type == "cuda":
            device_module.synchronize()
            device_module.reset_peak_memory_stats()

        start_time = time.time()
        total_tokens = 0

        for _ in self._progress_iterator(num_iterations, "Benchmark"):
            with torch.no_grad():
                generated = self._generate(input_ids, max_new_tokens)
                total_tokens += generated.shape[1] - input_ids.shape[1]

        if self.device.type == "cuda":
            device_module.synchronize()

        total_time = time.time() - start_time

        mem_alloc = device_module.max_memory_allocated() / 1e9 if self.device.type == "cuda" else 0.0
        mem_reserved = device_module.max_memory_reserved() / 1e9 if self.device.type == "cuda" else 0.0

        metrics = {
            "latency_ms": (total_time / num_iterations) * 1000,
            "throughput_tokens_per_sec": total_tokens / total_time,
            "memory_allocated_gb": mem_alloc,
            "memory_reserved_gb": mem_reserved,
        }

        self.log.info(f"Latency: {metrics['latency_ms']:.2f}ms, Throughput: {metrics['throughput_tokens_per_sec']:.2f} tok/s")
        return metrics

    def _generate(self, input_ids: torch.Tensor, max_new_tokens: int = 128) -> torch.Tensor:
        """Simple greedy/sampling generation."""
        # Get eos token
        eos_token_id = 2
        if self.tokenizer is not None:
            if hasattr(self.tokenizer, "eos_id"):
                eos_token_id = self.tokenizer.eos_id
            elif hasattr(self.tokenizer, "eos_token_id"):
                eos_token_id = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            attention_mask = self._create_attention_mask(input_ids)
            logits = self.model(input_ids, attention_masks=attention_mask)

            next_token_logits = logits[:, -1, :] / 0.7  # temperature
            topk_vals = torch.topk(next_token_logits, 50)[0][..., -1, None]
            next_token_logits[next_token_logits < topk_vals] = float("-inf")

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if (next_token == eos_token_id).all():
                break

        return input_ids

    def evaluate_computational_cost(self, seq_len: int = 512) -> dict[str, float]:
        """Evaluate computational cost (FLOPs, active parameters)."""
        self.log.info("Evaluating computational cost...")

        input_ids = torch.randint(0, self.model_args.vocab_size, (1, seq_len), device=self.device)
        total_params = sum(p.numel() for p in self.model.parameters())

        # Try theoretical FLOPs first
        total_flops, num_flops_per_token = FlopsCalculator.from_model_args(self.model, self.model_args, seq_len)

        # Try measured FLOPs if theoretical not available
        if total_flops is None:
            attention_mask = self._create_attention_mask(input_ids)
            measured = FlopsCalculator.measure_with_counter(self.model, input_ids, attention_mask)
            if measured is not None:
                total_flops = measured
                num_flops_per_token = measured / seq_len

        # Fallback to estimate
        if total_flops is None:
            total_flops = FlopsCalculator.estimate_from_params(total_params, seq_len)

        return {
            "total_flops": int(total_flops),
            "tflops": total_flops / 1e12,
            "active_params_billions": total_params / 1e9,
            "num_flops_per_token": num_flops_per_token,
            "gpu_peak_flops": self.gpu_peak_flops,
        }

    # -------------------------------------------------------------------------
    # Progress Helpers
    # -------------------------------------------------------------------------

    def _progress_iterator(self, count: int, desc: str):
        """Create a progress iterator that only shows on rank 0."""
        from tqdm import tqdm
        return tqdm(range(count), desc=desc, disable=self.rank != 0, unit="iter")

    def _run_with_progress(self, count: int, step_fn: Callable, desc: str):
        """Run a function multiple times with progress bar."""
        for _ in self._progress_iterator(count, desc):
            step_fn()

    # -------------------------------------------------------------------------
    # Full Evaluation Pipeline
    # -------------------------------------------------------------------------

    def run_full_evaluation(
        self,
        output_dir: str,
        skip_lm_eval: bool = False,
        lm_eval_only: bool = False,
        lm_eval_tasks: list[str] | None = None,
        lm_eval_limit: int | None = None,
    ) -> EvalResults:
        """Run full evaluation suite."""
        eval_start_time = time.time()
        results = EvalResults()

        phases = [
            ("1/4", "Routing efficiency", not lm_eval_only, lambda: self._eval_routing(results)),
            ("2/4", "Inference performance", not lm_eval_only, lambda: self._eval_performance(results)),
            ("3/4", "Computational cost", not lm_eval_only, lambda: self._eval_cost(results)),
            ("4/4", "lm_eval", not skip_lm_eval, lambda: self._eval_lm_eval(results, output_dir, lm_eval_tasks, lm_eval_limit)),
        ]

        for phase_num, phase_name, should_run, phase_fn in phases:
            if should_run:
                self.log.info(f"\n{'=' * 80}\n{phase_num} - Evaluating {phase_name}\n{'=' * 80}")
                phase_fn()
            else:
                self.log.info(f"\n{'=' * 80}\n{phase_num} - Skipping {phase_name}\n{'=' * 80}")

        # Finalize results
        results.walltime_seconds = time.time() - eval_start_time
        results.checkpoint_dir = str(self.checkpoint_dir)
        results.model_args = dataclasses.asdict(self.model_args)
        results.config = extract_config_for_results(self.job_config)

        # Save results (rank 0 only)
        if self.rank == 0:
            self._save_results(results, output_dir)

        if dist.is_initialized():
            dist.barrier()

        return results

    def _eval_routing(self, results: EvalResults):
        results.routing_stats = self.evaluate_routing_efficiency(num_samples=100, seq_len=512)

    def _eval_performance(self, results: EvalResults):
        perf = self.evaluate_inference_performance(max_new_tokens=128, num_warmup=3, num_iterations=10)
        results.latency_ms = perf["latency_ms"]
        results.throughput_tokens_per_sec = perf["throughput_tokens_per_sec"]
        results.memory_allocated_gb = perf["memory_allocated_gb"]
        results.memory_reserved_gb = perf["memory_reserved_gb"]

    def _eval_cost(self, results: EvalResults):
        cost = self.evaluate_computational_cost(seq_len=512)
        results.total_flops = cost["total_flops"]
        results.tflops = cost["tflops"]
        results.active_params_b = cost["active_params_billions"]
        results.num_flops_per_token = cost.get("num_flops_per_token")
        results.gpu_peak_flops = cost.get("gpu_peak_flops")

    def _eval_lm_eval(self, results: EvalResults, output_dir: str, tasks: list[str] | None, limit: int | None):
        results.lm_eval_results = self._run_lm_eval(
            output_dir=output_dir,
            tasks=tasks or ["mmlu", "hellaswag", "arc_easy"],
            limit=limit,
        )

    def _save_results(self, results: EvalResults, output_dir: str):
        """Save results to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_path / f"eval_results_{timestamp}.json"

        results.eval_timestamp = timestamp
        results.output_file = str(results_file.resolve())

        with open(results_file, "w") as f:
            json.dump(results.to_dict(), f, indent=2)

        self.log.info(f"Results saved to {results_file.resolve()}")
        self.printer.print_full_summary(results)

    # -------------------------------------------------------------------------
    # lm_eval Integration
    # -------------------------------------------------------------------------

    def _run_lm_eval(self, output_dir: str, tasks: list[str], limit: int | None = None) -> dict[str, Any] | None:
        """Run lm_eval directly on the torchtitan model."""
        import sys

        script_dir = Path(__file__).parent
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))

        try:
            from torchtitan_lm_eval import run_evaluation
        except ImportError as e:
            self.log.warning(f"lm_eval not available: {e}. Install with: pip install lm-eval")
            return None

        self.log.info(f"Running lm_eval on tasks: {tasks}")
        output_path = Path(output_dir) / "lm_eval_results.json"

        try:
            results = run_evaluation(
                tasks=tasks,
                batch_size=1,
                limit=limit,
                output_path=str(output_path) if self.rank == 0 else None,
                model=self.model,
                tokenizer=self.tokenizer,
                model_args=self.model_args,
            )

            if self.rank == 0:
                self.log.info("lm_eval completed successfully")
                for task_name, task_results in results.get("results", {}).items():
                    self.log.info(f"\n{task_name}:")
                    for metric, value in task_results.items():
                        if isinstance(value, float):
                            self.log.info(f"  {metric}: {value:.4f}")

                return results.get("results", {})
            return None

        except Exception as e:
            self.log.error(f"lm_eval failed: {e}")
            import traceback
            self.log.error(traceback.format_exc())
            return None

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def cleanup(self):
        """Cleanup distributed resources."""
        if dist.is_initialized():
            dist.destroy_process_group()


# =============================================================================
# Argument Parsing
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Distributed MoE Model Evaluation")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--config_file", type=str, required=True, help="Path to training config TOML file")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save results")
    parser.add_argument("--skip_lm_eval", action="store_true", help="Skip lm_eval benchmark")
    parser.add_argument("--lm_eval_only", action="store_true", help="Only run lm_eval")
    parser.add_argument("--lm_eval_tasks", type=str, nargs="+", default=["mmlu", "hellaswag", "arc_easy"])
    parser.add_argument("--lm_eval_limit", type=int, default=None, help="Limit examples per task")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    return parser.parse_args()


def get_output_dir(evaluator: MoEEvaluator, output_dir: str | None) -> str:
    """Determine output directory from args or config."""
    if output_dir is not None:
        return output_dir
    return str(Path(evaluator.job_config.job.dump_folder) / "eval")


# =============================================================================
# Main Entry Point
# =============================================================================


@record
def main():
    init_logger()
    args = parse_args()

    evaluator = None
    try:
        evaluator = MoEEvaluator(
            checkpoint_dir=args.checkpoint_dir,
            config_file=args.config_file,
            seed=args.seed,
        )
        output_dir = get_output_dir(evaluator, args.output_dir)

        if evaluator.rank == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        results = evaluator.run_full_evaluation(
            output_dir=output_dir,
            skip_lm_eval=args.skip_lm_eval,
            lm_eval_only=args.lm_eval_only,
            lm_eval_tasks=args.lm_eval_tasks,
            lm_eval_limit=args.lm_eval_limit,
        )

        if evaluator.rank == 0 and results.walltime_seconds:
            formatted = ResultsPrinter.format_walltime(results.walltime_seconds)
            evaluator.log.info(f"\nEvaluation complete! Walltime: {results.walltime_seconds:.2f}s ({formatted})")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
    finally:
        if evaluator is not None:
            evaluator.cleanup()
            if hasattr(evaluator, "model") and evaluator.model is not None:
                del evaluator.model
                if evaluator.device.type == "cuda":
                    device_module.empty_cache()


if __name__ == "__main__":
    main()

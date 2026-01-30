#!/usr/bin/env python3
"""
Comprehensive MoE Model Evaluation Script (Distributed)

Evaluates a trained MoE model across multiple dimensions:
1. Model accuracy (via lm_eval with HuggingFace conversion)
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
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

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
# Data Classes
# =============================================================================


@dataclass
class EvalResults:
    """Container for all evaluation results."""

    # Routing stats
    routing_stats: dict[str, Any] | None = None
    # Inference performance
    latency_ms: float | None = None
    throughput_tokens_per_sec: float | None = None
    memory_allocated_gb: float | None = None
    memory_reserved_gb: float | None = None
    # Computational cost
    total_flops: int | None = None
    tflops: float | None = None
    active_params_b: float | None = None
    num_flops_per_token: float | None = None
    gpu_peak_flops: float | None = None
    # Model accuracy (if lm_eval is run)
    lm_eval_results: dict[str, Any] | None = None
    # Walltime
    walltime_seconds: float | None = None
    # Config and model args for reproducibility
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
# Routing Metrics Calculation
# =============================================================================


def calculate_gini_coefficient(loads: list[float]) -> float:
    """Calculate Gini coefficient (0 = perfect equality, 1 = perfect inequality)."""
    sorted_loads = sorted(loads)
    n = len(sorted_loads)
    cumsum = sum((idx + 1) * val for idx, val in enumerate(sorted_loads))
    total = sum(sorted_loads)
    if total > 0:
        return (2 * cumsum) / (n * total) - (n + 1) / n
    return 0.0


def calculate_routing_metrics(tokens_per_expert) -> dict[str, Any]:
    """Calculate all routing metrics for a single MoE layer."""
    mean_load = float(tokens_per_expert.mean())
    std_load = float(tokens_per_expert.std())
    max_load = int(tokens_per_expert.max())
    min_load = int(tokens_per_expert.min())

    # Coefficient of variation (lower = better balance)
    cv = std_load / mean_load if mean_load > 0 else 0.0
    # Expert utilization rate
    num_experts = len(tokens_per_expert)
    active_experts = int((tokens_per_expert > 0).sum())
    utilization_rate = active_experts / num_experts
    # Gini coefficient
    gini = calculate_gini_coefficient(tokens_per_expert.tolist())

    return {
        "total_tokens_routed": int(tokens_per_expert.sum()),
        "num_experts": num_experts,
        "mean_tokens_per_expert": mean_load,
        "std_tokens_per_expert": std_load,
        "max_tokens_per_expert": max_load,
        "min_tokens_per_expert": min_load,
        "coefficient_of_variation": cv,
        "expert_utilization_rate": utilization_rate,
        "gini_coefficient": gini,
        "tokens_per_expert_distribution": tokens_per_expert.tolist(),
    }


def calculate_aggregate_routing_stats(layer_stats: dict[str, dict]) -> dict[str, Any]:
    """Calculate aggregate statistics across all MoE layers."""
    all_cvs = [s["coefficient_of_variation"] for s in layer_stats.values()]
    all_ginis = [s["gini_coefficient"] for s in layer_stats.values()]
    all_utils = [s["expert_utilization_rate"] for s in layer_stats.values()]

    return {
        "num_moe_layers": len(layer_stats),
        "avg_coefficient_of_variation": sum(all_cvs) / len(all_cvs) if all_cvs else 0,
        "avg_gini_coefficient": sum(all_ginis) / len(all_ginis) if all_ginis else 0,
        "avg_expert_utilization_rate": (
            sum(all_utils) / len(all_utils) if all_utils else 0
        ),
    }


# =============================================================================
# FLOPs Calculation
# =============================================================================


def get_theoretical_flops(
    model, model_args, seq_len: int
) -> tuple[int | None, float | None]:
    """Get theoretical FLOPs from model_args if available."""
    if not hasattr(model_args, "get_nparams_and_flops"):
        return None, None

    try:
        _, num_flops_per_token = model_args.get_nparams_and_flops(model, seq_len)
        total_flops = num_flops_per_token * seq_len
        logger.info(
            f"Theoretical FLOPs per token from model_args: {num_flops_per_token:.2e}"
        )
        return int(total_flops), num_flops_per_token
    except Exception as e:
        logger.warning(f"Could not get theoretical FLOPs from model_args: {e}")
        return None, None


def measure_flops_with_counter(model, input_ids, attention_mask) -> int | None:
    """Measure FLOPs using PyTorch's FlopCounterMode."""
    try:
        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            _ = model(input_ids, attention_masks=attention_mask)
        measured = flop_counter.get_total_flops()
        logger.info(f"Measured FLOPs via FlopCounterMode: {measured:.2e}")
        return measured
    except Exception as e:
        logger.warning(
            f"FlopCounterMode failed (likely due to flex_attention): {type(e).__name__}"
        )
        logger.info("Using theoretical FLOPs from model_args instead")
        return None


def estimate_flops_from_params(total_params: int, seq_len: int) -> int:
    """Rough FLOP estimate based on parameter count."""
    # Rough estimate: 2 * params * seq_len for forward pass
    total_flops = 2 * total_params * seq_len
    logger.warning(
        f"Using rough FLOP estimate based on parameter count: {total_flops / 1e12:.2f} TFLOPs"
    )
    return total_flops


# =============================================================================
# Input Preparation
# =============================================================================


def prepare_input_ids(
    tokenizer, prompts: list[str] | None, vocab_size: int, device
) -> torch.Tensor:
    """Prepare input token IDs from prompts or generate random tokens."""
    if tokenizer is None:
        logger.warning(
            "No tokenizer available. Using random token IDs for inference benchmark."
        )
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
        logger.info(f"Tokenized {len(prompts)} prompts, max length: {max_len}")
        return torch.tensor(padded, device=device)
    except (TypeError, AttributeError) as e:
        logger.warning(f"Tokenizer encode failed: {e}. Using random token IDs.")
        return torch.randint(0, vocab_size, (1, 64), device=device)


# =============================================================================
# Results Formatting
# =============================================================================


def format_walltime(seconds: float) -> str:
    """Format walltime as HH:MM:SS.ss."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


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
# Summary Printing
# =============================================================================


def print_routing_summary(routing_stats: dict | None):
    """Print routing efficiency summary."""
    print("\n[ROUTING EFFICIENCY]")
    print("-" * 80)
    if not routing_stats:
        print("  No routing data available")
        return

    agg = routing_stats.get("aggregate", {})
    print(
        f"  Average Coefficient of Variation: {agg.get('avg_coefficient_of_variation', 0):.4f}"
    )
    print(
        f"  Average Gini Coefficient:         {agg.get('avg_gini_coefficient', 0):.4f}"
    )
    print(
        f"  Average Expert Utilization:       {agg.get('avg_expert_utilization_rate', 0):.2%}"
    )
    print(f"  Number of MoE Layers:             {agg.get('num_moe_layers', 0)}")


def print_performance_summary(results: EvalResults):
    """Print inference performance summary."""
    print("\n[INFERENCE PERFORMANCE]")
    print("-" * 80)
    if results.latency_ms is None:
        print("  No inference performance data available")
        return

    print(f"  Latency:                          {results.latency_ms:.2f} ms")
    print(
        f"  Throughput:                       {results.throughput_tokens_per_sec:.2f} tokens/s"
    )
    print(f"  Memory Allocated:                 {results.memory_allocated_gb:.2f} GB")
    print(f"  Memory Reserved:                  {results.memory_reserved_gb:.2f} GB")


def print_cost_summary(results: EvalResults):
    """Print computational cost summary."""
    print("\n[COMPUTATIONAL COST]")
    print("-" * 80)
    if results.tflops is not None:
        print(f"  Total FLOPs:                      {results.tflops:.2f} TFLOPs")
    if results.active_params_b is not None:
        print(f"  Active Parameters:                {results.active_params_b:.2f}B")
    if results.num_flops_per_token:
        print(f"  FLOPs per Token:                  {results.num_flops_per_token:.2e}")
    if results.gpu_peak_flops:
        print(f"  GPU Peak FLOPs:                   {results.gpu_peak_flops:.2e}")


def print_lm_eval_summary(lm_eval_results: dict | None):
    """Print lm_eval results summary."""
    if not lm_eval_results:
        return

    print("\n[MODEL ACCURACY (lm_eval)]")
    print("-" * 80)
    for task, metrics in lm_eval_results.items():
        if isinstance(metrics, dict):
            print(f"  {task}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value}")


def print_walltime_summary(walltime_seconds: float | None):
    """Print walltime summary."""
    if walltime_seconds is None:
        return

    print("\n[WALLTIME]")
    print("-" * 80)
    print(
        f"  Total Time:                       {format_walltime(walltime_seconds)} ({walltime_seconds:.2f} seconds)"
    )


def print_output_summary(output_file: str | None):
    """Print output file location."""
    if output_file is None:
        return

    print("\n[OUTPUT]")
    print("-" * 80)
    print(f"  Results File: {output_file}")


def print_full_summary(results: EvalResults):
    """Print complete evaluation summary."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print_routing_summary(results.routing_stats)
    print_performance_summary(results)
    print_cost_summary(results)
    print_lm_eval_summary(results.lm_eval_results)
    print_walltime_summary(results.walltime_seconds)
    print_output_summary(results.output_file)

    print("\n" + "=" * 80)


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
        import torchtitan

        logger.info(
            f"torchtitan version: {torchtitan.__version__} "
            "(0.0.0 means __version__ is not defined correctly)."
        )

    def _load_config(self, config_file: str):
        """Load configuration from TOML file."""
        logger.info(f"Loading config from {config_file}")
        config_manager = ConfigManager()
        self.job_config = config_manager.parse_args(
            [f"--job.config_file={config_file}"]
        )
        logger.info(f"Starting evaluation: {self.job_config.job.description}")

        if self.job_config.experimental.custom_import:
            importlib.import_module(self.job_config.experimental.custom_import)

    def _init_environment(self, seed: int):
        """Initialize distributed or single-process environment."""
        self.is_distributed = "LOCAL_RANK" in os.environ

        if self.is_distributed:
            self._init_distributed_env()
        else:
            self._init_single_process_env(seed)

        logger.info(f"Set random seed to {seed} for reproducibility")

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
        logger.warning(
            "Running in single-process mode. For distributed GPU evaluation, use torchrun."
        )

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            device_module.set_device(self.device)
        else:
            self.device = torch.device("cpu")
            logger.warning("No CUDA available. Running on CPU (very slow).")

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
        ep, tp, pp = (
            p.expert_parallel_degree,
            p.tensor_parallel_degree,
            p.pipeline_parallel_degree,
        )
        min_required = ep * tp * pp

        if min_required > self.world_size:
            raise RuntimeError(
                f"Config requires {min_required} GPUs (EP={ep} x TP={tp} x PP={pp}), "
                f"but only {self.world_size} processes launched. "
                f"Use: torchrun --nproc_per_node={min_required} ..."
            )

        logger.info(
            f"Parallelism config: EP={ep}, TP={tp}, PP={pp}, world_size={self.world_size}"
        )

    def _init_model_args(self):
        """Initialize model arguments from train spec."""
        model_name = self.job_config.model.name
        model_flavor = self.job_config.model.flavor

        self.train_spec = get_train_spec(model_name)
        self.model_args = self.train_spec.model_args[model_flavor]
        self.model_args.update_from_config(self.job_config)

        logger.info(
            f"Building {model_name} {model_flavor} with "
            f"{json.dumps(dataclasses.asdict(self.model_args), indent=2, ensure_ascii=False)}"
        )

    def _init_device_monitor(self):
        """Initialize device memory monitor and GPU peak flops."""
        self.device_memory_monitor = build_device_memory_monitor()
        self.gpu_peak_flops = utils.get_peak_flops(
            self.device_memory_monitor.device_name
        )
        logger.info(f"Peak FLOPS used for computing MFU: {self.gpu_peak_flops:.3e}")

    def _init_tokenizer(self):
        """Initialize tokenizer from train spec."""
        self.tokenizer = (
            self.train_spec.build_tokenizer_fn(self.job_config)
            if self.train_spec.build_tokenizer_fn is not None
            else None
        )
        if self.tokenizer is not None:
            logger.info("Tokenizer loaded successfully")
        else:
            logger.warning("No tokenizer available for this model")

    # -------------------------------------------------------------------------
    # Model Loading
    # -------------------------------------------------------------------------

    def _load_model(self) -> torch.nn.Module:
        """Load model from checkpoint with parallelization."""
        logger.info(f"Loading model from {self.checkpoint_dir}")

        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {self.checkpoint_dir}"
            )

        model_dtype = TORCH_DTYPE_MAP[self.job_config.training.dtype]
        logger.info(f"Using dtype: {model_dtype}")
        # Create model on meta device
        with torch.device("meta"), utils.set_default_dtype(model_dtype):
            model = self.train_spec.model_cls(self.model_args)
        if self.is_distributed:
            model = self._load_distributed_model(model)
        else:
            model = self._load_single_process_model(model)
        model.eval()
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters (this rank): {total_params / 1e9:.2f}B")
        self._log_memory_usage("after checkpoint load")
        return model

    def _load_distributed_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Load model in distributed mode with parallelization."""
        model_converters = build_model_converters(self.job_config, self.parallel_dims)
        model_converters.convert(model)

        if self.parallel_dims.pp_enabled:
            raise NotImplementedError(
                "Pipeline parallel evaluation is not yet supported. "
                "Please use a config without pipeline parallelism for evaluation."
            )

        if self.train_spec.parallelize_fn is not None:
            self.train_spec.parallelize_fn(model, self.parallel_dims, self.job_config)

        init_device, buffer_device = self._get_init_devices()
        model.to_empty(device=init_device)
        with torch.no_grad():
            model.init_weights(buffer_device=buffer_device)
        model.train()  # Required for proper DTensor initialization

        self._log_memory_usage("after model init")
        self._load_checkpoint_dcp(model)
        return model

    def _load_single_process_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Load model in single-process mode."""
        logger.warning(
            "Loading model in single-process mode. "
            "For best results, use torchrun with matching parallelism."
        )

        model.to_empty(device=self.device)
        with torch.no_grad():
            model.init_weights()

        self._log_memory_usage("after model init")
        self._load_checkpoint_dcp(model)
        return model

    def _get_init_devices(self) -> tuple[str, str | None]:
        """Get initialization device and buffer device."""
        if self.job_config.training.enable_cpu_offload:
            return "cpu", device_type
        return device_type, None

    def _load_checkpoint_dcp(self, model: torch.nn.Module):
        """Load checkpoint using DCP."""
        logger.info("Loading checkpoint with DCP (this may take a few minutes)")
        state_dict = model.state_dict()
        dcp.load(state_dict, checkpoint_id=str(self.checkpoint_dir))
        model.load_state_dict(state_dict, assign=True)

    def _log_memory_usage(self, context: str):
        """Log GPU memory usage."""
        if self.device.type != "cuda":
            return
        stats = self.device_memory_monitor.get_peak_stats()
        logger.info(
            f"GPU memory {context}: {stats.max_reserved_gib:.2f}GiB ({stats.max_reserved_pct:.2f}%)"
        )

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

    def evaluate_routing_efficiency(
        self, num_samples: int = 100, seq_len: int = 512
    ) -> dict[str, Any]:
        """Evaluate MoE routing efficiency and load balance."""
        logger.info("Evaluating routing efficiency")

        self._reset_expert_counters()
        self._run_routing_inference(num_samples, seq_len)
        layer_stats = self._collect_layer_routing_stats()

        layer_stats["aggregate"] = calculate_aggregate_routing_stats(
            {k: v for k, v in layer_stats.items() if k != "aggregate"}
        )

        agg = layer_stats["aggregate"]
        logger.info(
            f"Aggregate routing: avg_cv={agg['avg_coefficient_of_variation']:.3f}, "
            f"avg_gini={agg['avg_gini_coefficient']:.3f}, "
            f"avg_util={agg['avg_expert_utilization_rate']:.2%}"
        )

        return layer_stats

    def _reset_expert_counters(self):
        """Reset expert token counters for all MoE layers."""
        for layer in self.model.layers.values():
            if hasattr(layer, "moe") and layer.moe is not None:
                layer.moe.tokens_per_expert.zero_()

    def _run_routing_inference(self, num_samples: int, seq_len: int):
        """Run inference to collect routing statistics."""
        with torch.no_grad():
            for _ in range(num_samples):
                input_ids = torch.randint(
                    0, self.model_args.vocab_size, (1, seq_len), device=self.device
                )
                attention_mask = self._create_attention_mask(input_ids)
                _ = self.model(input_ids, attention_masks=attention_mask)

    def _collect_layer_routing_stats(self) -> dict[str, Any]:
        """Collect routing statistics from all MoE layers."""
        stats = {}
        for i, layer in enumerate(self.model.layers.values()):
            if not (hasattr(layer, "moe") and layer.moe is not None):
                continue

            tokens_per_expert = layer.moe.tokens_per_expert.cpu().numpy()
            layer_metrics = calculate_routing_metrics(tokens_per_expert)
            stats[f"layer_{i}"] = layer_metrics

            logger.info(
                f"Layer {i} routing: mean={layer_metrics['mean_tokens_per_expert']:.1f}, "
                f"std={layer_metrics['std_tokens_per_expert']:.1f}, "
                f"cv={layer_metrics['coefficient_of_variation']:.3f}, "
                f"util={layer_metrics['expert_utilization_rate']:.2%}, "
                f"gini={layer_metrics['gini_coefficient']:.3f}"
            )

        return stats

    def evaluate_inference_performance(
        self,
        prompts: list[str] | None = None,
        max_new_tokens: int = 128,
        num_warmup: int = 3,
        num_iterations: int = 10,
    ) -> dict[str, float]:
        """Evaluate inference performance (latency, throughput, memory)."""
        logger.info("Evaluating inference performance")

        input_ids = prepare_input_ids(
            self.tokenizer, prompts, self.model_args.vocab_size, self.device
        )

        self._run_warmup(input_ids, num_warmup)
        metrics = self._run_benchmark(input_ids, max_new_tokens, num_iterations)

        logger.info(
            f"Latency: {metrics['latency_ms']:.2f}ms, "
            f"Throughput: {metrics['throughput_tokens_per_sec']:.2f} tokens/s, "
            f"Memory: {metrics['memory_allocated_gb']:.2f}GB allocated, "
            f"{metrics['memory_reserved_gb']:.2f}GB reserved"
        )

        return metrics

    def _run_warmup(self, input_ids: torch.Tensor, num_warmup: int):
        """Run warmup iterations."""
        logger.info(f"Warming up with {num_warmup} iterations")
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = self._generate(input_ids, max_new_tokens=32)

    def _run_benchmark(
        self, input_ids: torch.Tensor, max_new_tokens: int, num_iterations: int
    ) -> dict[str, float]:
        """Run benchmark iterations and collect metrics."""
        logger.info(f"Benchmarking with {num_iterations} iterations")

        if self.device.type == "cuda":
            device_module.synchronize()
            device_module.reset_peak_memory_stats()

        start_time = time.time()
        total_tokens = 0

        with torch.no_grad():
            for _ in range(num_iterations):
                generated = self._generate(input_ids, max_new_tokens=max_new_tokens)
                total_tokens += generated.shape[1] - input_ids.shape[1]

        if self.device.type == "cuda":
            device_module.synchronize()

        total_time = time.time() - start_time

        if self.device.type == "cuda":
            mem_alloc = device_module.max_memory_allocated() / 1e9
            mem_reserved = device_module.max_memory_reserved() / 1e9
        else:
            mem_alloc, mem_reserved = 0.0, 0.0

        return {
            "latency_ms": (total_time / num_iterations) * 1000,
            "throughput_tokens_per_sec": total_tokens / total_time,
            "memory_allocated_gb": mem_alloc,
            "memory_reserved_gb": mem_reserved,
        }

    def _generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Simple greedy/sampling generation."""
        eos_token_id = (
            self.tokenizer.eos_token_id
            if self.tokenizer is not None and hasattr(self.tokenizer, "eos_token_id")
            else 2
        )

        for _ in range(max_new_tokens):
            attention_mask = self._create_attention_mask(input_ids)
            logits = self.model(input_ids, attention_masks=attention_mask)

            next_token_logits = logits[:, -1, :] / temperature

            if top_k > 0:
                topk_vals = torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[next_token_logits < topk_vals] = float("-inf")

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if (next_token == eos_token_id).all():
                break

        return input_ids

    def evaluate_computational_cost(self, seq_len: int = 512) -> dict[str, float]:
        """Evaluate computational cost (FLOPs, active parameters)."""
        logger.info("Evaluating computational cost")

        input_ids = torch.randint(
            0, self.model_args.vocab_size, (1, seq_len), device=self.device
        )
        total_params = sum(p.numel() for p in self.model.parameters())

        # Try theoretical FLOPs first
        total_flops, num_flops_per_token = get_theoretical_flops(
            self.model, self.model_args, seq_len
        )

        # Try measured FLOPs if theoretical not available
        if total_flops is None:
            attention_mask = self._create_attention_mask(input_ids)
            measured = measure_flops_with_counter(self.model, input_ids, attention_mask)
            if measured is not None:
                total_flops = measured
                num_flops_per_token = measured / seq_len

        # Fallback to estimate
        if total_flops is None:
            total_flops = estimate_flops_from_params(total_params, seq_len)

        tflops = total_flops / 1e12
        active_params_b = total_params / 1e9

        logger.info(
            f"Total FLOPs: {tflops:.2f} TFLOPs, "
            f"Active params: {active_params_b:.2f}B, "
            f"Peak GPU FLOPS: {self.gpu_peak_flops:.2e}"
        )

        return {
            "total_flops": int(total_flops),
            "tflops": tflops,
            "active_params_billions": active_params_b,
            "num_flops_per_token": num_flops_per_token,
            "gpu_peak_flops": self.gpu_peak_flops,
        }

    # -------------------------------------------------------------------------
    # Full Evaluation Pipeline
    # -------------------------------------------------------------------------

    def run_full_evaluation(
        self,
        output_dir: str,
        skip_lm_eval: bool = False,
        lm_eval_tasks: list[str] | None = None,
    ) -> EvalResults:
        """Run full evaluation suite."""
        eval_start_time = time.time()
        logger.info("Starting full evaluation")

        results = EvalResults()

        # 1. Routing efficiency
        self._log_phase("1/4 - Evaluating routing efficiency")
        results.routing_stats = self.evaluate_routing_efficiency(
            num_samples=100, seq_len=512
        )

        # 2. Inference performance
        self._log_phase("2/4 - Evaluating inference performance")
        perf = self.evaluate_inference_performance(
            max_new_tokens=128, num_warmup=3, num_iterations=10
        )
        results.latency_ms = perf["latency_ms"]
        results.throughput_tokens_per_sec = perf["throughput_tokens_per_sec"]
        results.memory_allocated_gb = perf["memory_allocated_gb"]
        results.memory_reserved_gb = perf["memory_reserved_gb"]

        # 3. Computational cost
        self._log_phase("3/4 - Evaluating computational cost")
        cost = self.evaluate_computational_cost(seq_len=512)
        results.total_flops = cost["total_flops"]
        results.tflops = cost["tflops"]
        results.active_params_b = cost["active_params_billions"]
        results.num_flops_per_token = cost.get("num_flops_per_token")
        results.gpu_peak_flops = cost.get("gpu_peak_flops")

        # 4. Model accuracy (lm_eval)
        if skip_lm_eval:
            self._log_phase("4/4 - Skipping lm_eval (--skip_lm_eval flag set)")
        else:
            self._log_phase("4/4 - Running lm_eval (this may take a while)")
            if self.rank == 0:
                results.lm_eval_results = self._run_lm_eval(
                    output_dir=output_dir, tasks=lm_eval_tasks or ["mmlu", "hellaswag"]
                )

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

    def _log_phase(self, message: str):
        """Log evaluation phase."""
        logger.info("\n" + "=" * 80)
        logger.info(message)
        logger.info("=" * 80)

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

        logger.info(f"\nResults saved to {results_file.resolve()}")
        print_full_summary(results)

    # -------------------------------------------------------------------------
    # lm_eval Integration
    # -------------------------------------------------------------------------

    def _run_lm_eval(self, output_dir: str, tasks: list[str]) -> dict[str, Any] | None:
        """Run lm_eval by converting checkpoint to HF format."""
        hf_checkpoint_dir = Path(output_dir) / "hf_checkpoint"

        if not self._convert_to_hf(hf_checkpoint_dir):
            return None

        return self._run_lm_eval_command(hf_checkpoint_dir, output_dir, tasks)

    def _convert_to_hf(self, hf_checkpoint_dir: Path) -> bool:
        """Convert checkpoint to HuggingFace format."""
        logger.info("Converting checkpoint to HuggingFace format")

        cmd = [
            "python",
            "./scripts/checkpoint_conversion/convert_to_hf.py",
            str(self.checkpoint_dir),
            str(hf_checkpoint_dir),
            "--model_name",
            self.job_config.model.name,
            "--model_flavor",
            self.job_config.model.flavor,
            "--export_dtype",
            "bfloat16",
        ]

        if self.job_config.model.hf_assets_path:
            cmd.extend(["--hf_assets_path", self.job_config.model.hf_assets_path])

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"HF checkpoint saved to {hf_checkpoint_dir}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to convert checkpoint: {e.stderr}")
            return False

    def _run_lm_eval_command(
        self, hf_checkpoint_dir: Path, output_dir: str, tasks: list[str]
    ) -> dict[str, Any] | None:
        """Run lm_eval command on converted checkpoint."""
        logger.info(f"Running lm_eval on tasks: {tasks}")

        if shutil.which("lm_eval") is None:
            logger.warning(
                "lm_eval not found. Please install it:\n"
                "  pip install lm-eval\n"
                f"Then run manually:\n"
                f"  lm_eval --model hf --model_args pretrained={hf_checkpoint_dir},dtype=auto "
                f"--tasks {','.join(tasks)} --batch_size auto"
            )
            return None

        try:
            result = subprocess.run(
                [
                    "lm_eval",
                    "--model",
                    "hf",
                    "--model_args",
                    f"pretrained={hf_checkpoint_dir},dtype=auto",
                    "--tasks",
                    ",".join(tasks),
                    "--batch_size",
                    "auto",
                    "--output_path",
                    str(Path(output_dir) / "lm_eval_output"),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("lm_eval completed successfully")
            logger.info(result.stdout)

            output_file = Path(output_dir) / "lm_eval_output" / "results.json"
            if output_file.exists():
                with open(output_file) as f:
                    return json.load(f)

        except subprocess.CalledProcessError as e:
            logger.error(f"lm_eval failed: {e.stderr}")
        except FileNotFoundError:
            logger.warning("lm_eval command not found. Skipping.")

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
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to checkpoint directory (DCP format)",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to training config TOML file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results (default: {dump_folder}/eval)",
    )
    parser.add_argument(
        "--skip_lm_eval",
        action="store_true",
        help="Skip lm_eval (requires separate installation)",
    )
    parser.add_argument(
        "--lm_eval_tasks",
        type=str,
        nargs="+",
        default=["mmlu", "hellaswag", "arc_easy"],
        help="Tasks to run for lm_eval",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for reproducibility (default: 1337)",
    )
    return parser.parse_args()


def get_output_dir(evaluator: MoEEvaluator, output_dir: str | None) -> str:
    """Determine output directory from args or config."""
    if output_dir is not None:
        return output_dir
    default_dir = str(Path(evaluator.job_config.job.dump_folder) / "eval")
    logger.info(f"Using default output directory: {default_dir}")
    return default_dir


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
            lm_eval_tasks=args.lm_eval_tasks,
        )
        if results.walltime_seconds is not None:
            logger.info(f"\nEvaluation complete!")
            logger.info(
                f"Total walltime: {format_walltime(results.walltime_seconds)} "
                f"({results.walltime_seconds:.2f} seconds)"
            )
        else:
            logger.info("\nEvaluation complete!")

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
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

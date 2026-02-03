# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Custom lm_eval model wrapper for torchtitan models.

This allows running lm_eval benchmarks directly on torchtitan models
without converting to HuggingFace format.

Usage:
    from scripts.eval.torchtitan_lm_eval import TorchTitanLM

    model = TorchTitanLM(
        checkpoint_dir="/path/to/checkpoint",
        config_file="/path/to/config.toml",
    )

    results = lm_eval.simple_evaluate(
        model=model,
        tasks=["hellaswag", "arc_easy"],
        batch_size=4,
    )
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from transformers import AutoTokenizer

import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.config import JobConfig
from torchtitan.models.attention import create_attention_mask, get_causal_mask_mod

try:
    from lm_eval.api.instance import Instance
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
except ImportError:
    raise ImportError(
        "lm_eval is required for this module. Install with: pip install lm-eval"
    )


# =============================================================================
# Rank-Aware Logging
# =============================================================================


class RankLogger:
    """Logger that only outputs on rank 0."""

    def __init__(self, name: str, rank: int):
        self._logger = logging.getLogger(name)
        self._rank = rank

    def info(self, msg: str):
        if self._rank == 0:
            self._logger.info(msg)

    def warning(self, msg: str):
        if self._rank == 0:
            self._logger.warning(msg)

    def error(self, msg: str):
        self._logger.error(msg)  # Errors on all ranks

    def debug(self, msg: str):
        if self._rank == 0:
            self._logger.debug(msg)


# =============================================================================
# TorchTitan LM Wrapper
# =============================================================================


@register_model("torchtitan")
class TorchTitanLM(LM):
    """
    lm_eval model wrapper for torchtitan models.

    This wrapper allows running lm_eval benchmarks directly on torchtitan
    checkpoints without converting to HuggingFace format.
    """

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        config_file: Optional[str] = None,
        batch_size: int = 1,
        max_length: Optional[int] = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        # Optional: pass existing model and tokenizer to avoid reloading
        model: Optional[torch.nn.Module] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        model_args: Optional[object] = None,
    ):
        super().__init__()

        self._batch_size = batch_size
        self._device = torch.device(device)
        self._dtype = getattr(torch, dtype)

        # Set up distributed info
        self._is_distributed = dist.is_initialized()
        self._rank = dist.get_rank() if self._is_distributed else 0
        self._world_size = dist.get_world_size() if self._is_distributed else 1

        # Initialize rank-aware logger
        self._log = RankLogger(__name__, self._rank)

        if model is not None and tokenizer is not None:
            self._init_from_existing(model, tokenizer, model_args, max_length)
        else:
            self._init_from_checkpoint(checkpoint_dir, config_file, max_length)

        self._setup_tokenizer()
        self._setup_accelerator()

    def _init_from_existing(self, model, tokenizer, model_args, max_length):
        """Initialize from provided model and tokenizer."""
        self.model = model
        self.tokenizer = tokenizer
        self.model_args = model_args
        self._max_length = max_length or getattr(model_args, "max_seq_len", 4096) if model_args else 4096

    def _init_from_checkpoint(self, checkpoint_dir, config_file, max_length):
        """Initialize by loading from checkpoint."""
        if checkpoint_dir is None or config_file is None:
            raise ValueError("Either provide (model, tokenizer) or (checkpoint_dir, config_file)")

        self.checkpoint_dir = Path(checkpoint_dir)
        self.config_file = config_file

        # Load config
        self.job_config = JobConfig()
        self.job_config.parse_args([f"--job.config_file={config_file}"])

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.job_config.model.hf_assets_path,
            local_files_only=True,
        )

        # Load model
        self._load_model()
        self._max_length = max_length or getattr(self.model_args, "max_seq_len", 4096)

    def _setup_tokenizer(self):
        """Set up tokenizer attributes."""
        if hasattr(self.tokenizer, "pad_token") and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Detect eos_id
        if hasattr(self.tokenizer, "eos_id"):
            self._eos_id = self.tokenizer.eos_id
        elif hasattr(self.tokenizer, "eos_token_id"):
            self._eos_id = self.tokenizer.eos_token_id
        else:
            self._eos_id = 2

    def _setup_accelerator(self):
        """Set up mock accelerator for lm_eval compatibility."""
        self.accelerator = self._MockAccelerator(self._rank, self._world_size, self._is_distributed)

    class _MockAccelerator:
        """Mock accelerator for lm_eval compatibility."""

        def __init__(self, rank: int, world_size: int, is_distributed: bool):
            self.rank = rank
            self.world_size = world_size
            self.is_main_process = rank == 0
            self.num_processes = world_size
            self._is_distributed = is_distributed

        def gather(self, obj):
            """Gather objects from all ranks."""
            if not self._is_distributed or self.world_size == 1:
                return obj.unsqueeze(0) if isinstance(obj, torch.Tensor) else [obj]

            gathered = [None] * self.world_size
            dist.all_gather_object(gathered, obj)

            if isinstance(obj, torch.Tensor):
                return torch.stack([t.cpu() for t in gathered])
            return gathered

        def wait_for_everyone(self):
            """Synchronize all processes."""
            if self._is_distributed:
                dist.barrier()

    def _load_model(self):
        """Load the torchtitan model from checkpoint."""
        train_spec = train_spec_module.get_train_spec(self.job_config.model.name)
        self.model_args = train_spec.model_args[self.job_config.model.flavor]

        with torch.device(self._device):
            self.model = train_spec.model_cls(self.model_args)

        self.model = self.model.to(self._dtype)
        self.model.eval()

        # Load checkpoint
        model_wrapper = ModelWrapper(self.model)
        state_dict = model_wrapper._get_state_dict()
        dcp.load(state_dict, checkpoint_id=str(self.checkpoint_dir))

        self.model.load_state_dict({k: v for k, v in state_dict.items()}, strict=False)

    # -------------------------------------------------------------------------
    # LM Properties
    # -------------------------------------------------------------------------

    @property
    def eot_token_id(self) -> int:
        return self._eos_id

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    # -------------------------------------------------------------------------
    # Tokenization
    # -------------------------------------------------------------------------

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        """Encode string to token IDs."""
        if hasattr(self.tokenizer, "eos_id"):
            return self.tokenizer.encode(string, add_bos=False, add_eos=False)
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: List[int], **kwargs) -> str:
        """Decode token IDs to string."""
        return self.tokenizer.decode(tokens)

    # -------------------------------------------------------------------------
    # Model Forward
    # -------------------------------------------------------------------------

    def _create_attention_mask(self, input_ids: torch.Tensor):
        """Create attention mask for model forward pass."""
        attn_type = getattr(self.model_args, "attn_type", "sdpa") if self.model_args else "sdpa"
        if attn_type not in ["flex", "varlen"]:
            return None

        B, seq_len = input_ids.shape
        mask_mod = get_causal_mask_mod()
        return create_attention_mask(mask_mod, B, None, seq_len, seq_len)

    def _model_call(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run model forward pass and return logits."""
        attention_mask = self._create_attention_mask(input_ids)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_masks=attention_mask)

            if isinstance(outputs, tuple):
                return outputs[0]
            if hasattr(outputs, "logits"):
                return outputs.logits
            return outputs

    def _model_generate(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
        stop_tokens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively (batch_size=1 only)."""
        if input_ids.shape[0] != 1:
            raise ValueError(f"_model_generate only supports batch_size=1, got {input_ids.shape[0]}")

        generated = input_ids.clone()
        stop_tokens = stop_tokens or [self.eot_token_id]

        with torch.no_grad():
            for _ in range(max_tokens):
                logits = self._model_call(generated)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)

                if next_token.item() in stop_tokens:
                    break

        return generated

    # -------------------------------------------------------------------------
    # LM Eval Methods
    # -------------------------------------------------------------------------

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Compute log-likelihood of continuations given contexts."""
        from tqdm import tqdm

        results = []
        total = len(requests)

        self._log.info(f"Processing {total} loglikelihood requests...")

        iterator = tqdm(requests, desc="loglikelihood", disable=self._rank != 0, unit="req")

        for request in iterator:
            context, continuation = request.args
            context_ids = self.tok_encode(context)
            continuation_ids = self.tok_encode(continuation)

            input_ids = context_ids + continuation_ids
            input_tensor = torch.tensor([input_ids], device=self._device)

            logits = self._model_call(input_tensor)
            log_probs = F.log_softmax(logits, dim=-1)

            continuation_start = len(context_ids)
            total_log_prob = 0.0
            is_greedy = True

            for i, token_id in enumerate(continuation_ids):
                pos = continuation_start + i - 1
                if 0 <= pos < log_probs.shape[1]:
                    total_log_prob += log_probs[0, pos, token_id].item()
                    if torch.argmax(log_probs[0, pos]).item() != token_id:
                        is_greedy = False

            results.append((total_log_prob, is_greedy))

        self._log.info(f"Completed {total} loglikelihood requests")
        return results

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Compute rolling log-likelihood (perplexity) over a string."""
        from tqdm import tqdm

        results = []
        total = len(requests)

        self._log.info(f"Processing {total} loglikelihood_rolling requests...")

        iterator = tqdm(requests, desc="loglikelihood_rolling", disable=self._rank != 0, unit="req")

        for request in iterator:
            (text,) = request.args
            input_ids = self.tok_encode(text)

            if not input_ids:
                results.append((0.0, True))
                continue

            if len(input_ids) > self._max_length:
                input_ids = input_ids[-self._max_length:]

            input_tensor = torch.tensor([input_ids], device=self._device)
            logits = self._model_call(input_tensor)
            log_probs = F.log_softmax(logits, dim=-1)

            total_log_prob = sum(
                log_probs[0, i - 1, input_ids[i]].item()
                for i in range(1, len(input_ids))
            )

            results.append((total_log_prob, True))

        self._log.info(f"Completed {total} loglikelihood_rolling requests")
        return results

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate text until stop sequences are reached."""
        from tqdm import tqdm

        results = []
        total = len(requests)

        self._log.info(f"Processing {total} generate_until requests...")

        default_eos = getattr(self.tokenizer, "eos_token", "</s>")

        iterator = tqdm(requests, desc="generate_until", disable=self._rank != 0, unit="req")

        for request in iterator:
            context = request.args[0]
            gen_kwargs = request.kwargs if hasattr(request, "kwargs") else {}

            until = gen_kwargs.get("until", [default_eos])
            max_gen_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks)

            context_ids = self.tok_encode(context)
            input_tensor = torch.tensor([context_ids], device=self._device)

            # Encode stop tokens
            stop_tokens = []
            for s in until:
                if s:
                    encoded = self.tok_encode(s)
                    if encoded:
                        stop_tokens.append(encoded[0])

            output_ids = self._model_generate(input_tensor, max_gen_toks, stop_tokens)

            generated_ids = output_ids[0, len(context_ids):].tolist()
            generated_text = self.tok_decode(generated_ids)

            # Truncate at stop sequences
            for stop_seq in until:
                if stop_seq in generated_text:
                    generated_text = generated_text[: generated_text.index(stop_seq)]

            results.append(generated_text)

        self._log.info(f"Completed {total} generate_until requests")
        return results


# =============================================================================
# Evaluation Entry Point
# =============================================================================


def run_evaluation(
    checkpoint_dir: Optional[str] = None,
    config_file: Optional[str] = None,
    tasks: Optional[List[str]] = None,
    batch_size: int = 1,
    num_fewshot: Optional[int] = None,
    limit: Optional[int] = None,
    output_path: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    model_args: Optional[object] = None,
) -> dict:
    """
    Run lm_eval benchmarks on a torchtitan model.

    For distributed (FSDP) models, all ranks run lm_eval together since FSDP
    requires all ranks to participate in forward passes.
    """
    import json

    import lm_eval

    tasks = tasks or ["hellaswag"]

    # Create model wrapper
    if model is not None and tokenizer is not None:
        lm_wrapper = TorchTitanLM(model=model, tokenizer=tokenizer, model_args=model_args, batch_size=batch_size)
    else:
        lm_wrapper = TorchTitanLM(checkpoint_dir=checkpoint_dir, config_file=config_file, batch_size=batch_size)

    # Run evaluation
    results = lm_eval.simple_evaluate(
        model=lm_wrapper,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        batch_size=batch_size,
    )

    # Save results (rank 0 only)
    is_rank_0 = not lm_wrapper._is_distributed or lm_wrapper._rank == 0
    if output_path and is_rank_0:
        serializable = {k: v for k, v in results.items() if k != "samples"}
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)

    return results if is_rank_0 else {}


# =============================================================================
# CLI Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run lm_eval on a torchtitan model")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--tasks", type=str, nargs="+", default=["hellaswag"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    results = run_evaluation(
        checkpoint_dir=args.checkpoint_dir,
        config_file=args.config_file,
        tasks=args.tasks,
        batch_size=args.batch_size,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        output_path=args.output,
    )

    # Print summary (rank 0 only)
    if results:
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        for task_name, task_results in results.get("results", {}).items():
            print(f"\n{task_name}:")
            for metric, value in task_results.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")

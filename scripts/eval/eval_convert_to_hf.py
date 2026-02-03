#!/usr/bin/env python3
"""
HuggingFace checkpoint conversion for evaluation.

This module extends the base convert_to_hf functionality to also generate
a proper HuggingFace config.json file required for lm_eval compatibility.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
import torchtitan.protocols.train_spec as train_spec_module
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.config import TORCH_DTYPE_MAP


# Mapping from torchtitan model names to HuggingFace model_type
MODEL_NAME_TO_HF_TYPE = {
    "deepseek_v3": "deepseek_v3",
    "llama3": "llama",
    "llama4": "llama4",
    "qwen3": "qwen3",
}


def generate_hf_config(model_name: str, model_args, export_dtype: str) -> dict:
    """Generate a HuggingFace-compatible config.json based on model args."""
    model_type = MODEL_NAME_TO_HF_TYPE.get(model_name, model_name)

    # Map export dtype to torch_dtype string
    dtype_map = {
        "float16": "float16",
        "bfloat16": "bfloat16",
        "float32": "float32",
    }

    if model_name == "deepseek_v3":
        # DeepSeek V3 specific config
        moe_args = model_args.moe_args
        config = {
            "architectures": ["DeepseekV3ForCausalLM"],
            "model_type": model_type,
            "vocab_size": model_args.vocab_size,
            "hidden_size": model_args.dim,
            "intermediate_size": model_args.inter_dim,
            "moe_intermediate_size": model_args.moe_inter_dim,
            "num_hidden_layers": model_args.n_layers,
            "num_attention_heads": model_args.n_heads,
            "num_key_value_heads": model_args.n_heads,  # MLA uses same as num_heads
            "hidden_act": "silu",
            "max_position_embeddings": model_args.max_seq_len,
            "rms_norm_eps": model_args.norm_eps,
            "tie_word_embeddings": False,
            "rope_theta": model_args.rope_theta,
            "attention_bias": False,
            "torch_dtype": dtype_map.get(export_dtype, "bfloat16"),
            # MoE specific
            "n_routed_experts": moe_args.num_experts,
            "n_shared_experts": moe_args.num_shared_experts,
            "num_experts_per_tok": moe_args.top_k,
            "first_k_dense_replace": model_args.n_dense_layers,
            "routed_scaling_factor": moe_args.route_scale,
            "scoring_func": moe_args.score_func,
            # MLA specific
            "q_lora_rank": model_args.q_lora_rank,
            "kv_lora_rank": model_args.kv_lora_rank,
            "qk_nope_head_dim": model_args.qk_nope_head_dim,
            "qk_rope_head_dim": model_args.qk_rope_head_dim,
            "v_head_dim": model_args.v_head_dim,
        }
    elif model_name in ("llama3", "llama4"):
        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": model_type,
            "vocab_size": model_args.vocab_size,
            "hidden_size": model_args.dim,
            "intermediate_size": (
                model_args.intermediate_dim
                if hasattr(model_args, "intermediate_dim")
                else model_args.dim * 4
            ),
            "num_hidden_layers": model_args.n_layers,
            "num_attention_heads": model_args.n_heads,
            "num_key_value_heads": getattr(
                model_args, "n_kv_heads", model_args.n_heads
            ),
            "hidden_act": "silu",
            "max_position_embeddings": model_args.max_seq_len,
            "rms_norm_eps": model_args.norm_eps,
            "tie_word_embeddings": False,
            "rope_theta": getattr(model_args, "rope_theta", 10000.0),
            "attention_bias": False,
            "torch_dtype": dtype_map.get(export_dtype, "bfloat16"),
        }
    else:
        # Generic config for other models
        config = {
            "model_type": model_type,
            "vocab_size": getattr(model_args, "vocab_size", 32000),
            "hidden_size": getattr(model_args, "dim", 4096),
            "num_hidden_layers": getattr(model_args, "n_layers", 32),
            "num_attention_heads": getattr(model_args, "n_heads", 32),
            "torch_dtype": dtype_map.get(export_dtype, "bfloat16"),
        }

    return config


def copy_tokenizer_files(hf_assets_path: Path, output_dir: Path):
    """Copy tokenizer files from assets path to output directory."""
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]

    for filename in tokenizer_files:
        src = hf_assets_path / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)


@torch.inference_mode()
def convert_to_hf(
    input_dir,
    output_dir,
    model_name,
    model_flavor,
    hf_assets_path,
    export_dtype,
):
    """
    Convert a torchtitan checkpoint to HuggingFace format.

    This function:
    1. Loads the model and converts state dict to HF format
    2. Saves the model weights as safetensors
    3. Generates a config.json with proper model_type for HF compatibility
    4. Copies tokenizer files from assets path
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load model and model args so that we can get the state dict shape
    train_spec = train_spec_module.get_train_spec(model_name)
    model_args = train_spec.model_args[model_flavor]

    with torch.device("cpu"):
        model = train_spec.model_cls(model_args)
    model = ModelWrapper(model)

    # pyrefly: ignore[bad-instantiation, not-callable]
    sd_adapter = train_spec.state_dict_adapter(model_args, hf_assets_path)
    assert (
        sd_adapter is not None
    ), "trying to convert checkpoint from DCP to HF safetensors format, but sd_adapter is not provided."

    # allocate state dict memory with empty weights to load checkpoint
    state_dict = model._get_state_dict()
    dcp.load(
        state_dict,
        checkpoint_id=str(input_dir),
    )

    # convert state dict tt->hf
    hf_state_dict = sd_adapter.to_hf(state_dict)

    storage_writer = HuggingFaceStorageWriter(
        path=output_dir,
        save_distributed=True,
        fqn_to_index_mapping=sd_adapter.fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )

    # map and apply export dtype if needed
    target_dtype = TORCH_DTYPE_MAP[export_dtype]
    if target_dtype != torch.float32:
        hf_state_dict = {k: v.to(target_dtype) for k, v in hf_state_dict.items()}

    dcp.save(
        hf_state_dict,
        storage_writer=storage_writer,
    )

    # Generate and save HuggingFace config.json
    hf_config = generate_hf_config(model_name, model_args, export_dtype)
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(hf_config, f, indent=2)

    # Copy tokenizer files from assets path
    if hf_assets_path:
        copy_tokenizer_files(Path(hf_assets_path), output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DCP weights to HF format.")
    parser.add_argument(
        "input_dir", type=Path, help="Input directory with DCP weights."
    )
    parser.add_argument(
        "output_dir", type=Path, help="Output directory for HF checkpoint."
    )
    parser.add_argument(
        "--hf_assets_path",
        type=Path,
        help="Path to HF assets directory. This is used to get the model.safetensors.index.json mapping",
        default="./assets/hf/Llama-3.1-8B",
    )
    parser.add_argument("--model_name", type=str, nargs="?", default="llama3")
    parser.add_argument("--model_flavor", type=str, nargs="?", default="8B")
    parser.add_argument(
        "--export_dtype",
        type=str,
        nargs="?",
        choices=["float16", "bfloat16", "float32"],
        default="float32",
        help="Export dtype for HF checkpoint (default: float32)",
    )
    args = parser.parse_args()

    convert_to_hf(
        args.input_dir,
        args.output_dir,
        args.model_name,
        args.model_flavor,
        args.hf_assets_path,
        args.export_dtype,
    )

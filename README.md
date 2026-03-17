<div align="center">

# torchtitan

#### A PyTorch native platform for training generative AI models

[![8 GPU Feature Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_features.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_features.yaml?query=branch%3Amain)
[![8 GPU Model Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_models.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_models.yaml?query=branch%3Amain)
[![arXiv](https://img.shields.io/badge/arXiv-2410.06511-b31b1b.svg)](https://arxiv.org/abs/2410.06511)
[![ICLR](https://img.shields.io/badge/ICLR-2025-violet.svg)](https://iclr.cc/virtual/2025/poster/29620)
[![forum](https://img.shields.io/badge/pytorch-forum-DE3412.svg)](https://discuss.pytorch.org/c/distributed/torchtitan/44)
[![license](https://img.shields.io/badge/license-BSD_3--Clause-lightgrey.svg)](./LICENSE)
[![pip](https://img.shields.io/pypi/v/torchtitan?color=blue)](https://pypi.org/project/torchtitan/)
[![conda](https://img.shields.io/conda/vn/conda-forge/torchtitan?color=green)](https://anaconda.org/conda-forge/torchtitan)


</div>

`torchtitan` is under extensive development. To use the latest features of `torchtitan`, we recommend using the most recent PyTorch nightly.

~~~~~~~~~~~~
## torchtitan_moe fork change 1 start
~~~~~~~~~~~~

## Setup
   1. Make sure you have `uv` installed
      - https://docs.astral.sh/uv/getting-started/installation/#installation-methods
   2. Setup a virtual environment
      - `python3.10 -m venv <directory>`
      - example: `python3.10 -m venv .venv`
      - using uv: `uv venv .venv`
   3. Activate the virtual environment
      - Windows cmd: `.venv\Scripts\activate.bat`
      - Windows powershell: `.venv\Scripts\Activate.ps1`
      - Linux/Mac/Other: `source .venv/bin/activate`
   4. Pip install all dependencies
      - `uv sync && uv pip install --pre "torch==2.11.0.dev20260129+cu126" --index-url https://download.pytorch.org/whl/nightly/cu126 --force-reinstall`

## Command to run training script (single node - up to 8 GPUs)
   1. [Downloading a tokenizer](#downloading-a-tokenizer)
      - `python scripts/download_hf_assets.py --repo_id deepseek-ai/deepseek-moe-16b-base --assets tokenizer --hf_token=$YOUR_HF_TOKEN`
   2. [Start a training run](#start-a-training-run)
      - `LOG_RANK=0 NGPU=4 CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b_nvidia_4x_a100_80GBmem.toml" ./run_train.sh`



## Command to run training script (multi-node)
1. [Multi-Node Training](#multi-node-training)

## Evaluation
- See [`scripts/eval/EVAL_README.md`](scripts/eval/EVAL_README.md) for full documentation

## Mixture of Depths (MoD)
MoD ([arxiv.org/abs/2404.02258](https://arxiv.org/abs/2404.02258)) reduces FLOPs by only processing a fraction of tokens through MoE layers. To enable, add to your TOML config:
```toml
[parallelism]
mod_capacity_ratio = 0.125  # process 12.5% of tokens; 0.0 = disabled (default)
```

## Residual Routing
Standard MoE routes tokens based on the full hidden state, which converges across tokens in deep layers — causing routing collapse (a few experts receive most tokens while others starve). Residual routing fixes this by routing on the **attention residual** (what the attention layer contributed) instead. This signal stays diverse at all depths because different tokens need different refinements, even when their accumulated representations have converged. To enable, add to your TOML config:
```toml
[parallelism]
residual_routing = true  # false = standard routing (default)
```

## Evals done so far

### MoE baseline vs MoD ([arxiv.org/abs/2404.02258])
```

python3 scripts/eval/compare_eval_results.py results/deepseek_16B_4xa100_moe_baseline_20260313_012550/summary.json results/deepseek_16B_mod_4xa100_20260316_105118/summary.json

==================================================================================
                             MoE EVALUATION COMPARISON
==================================================================================

┌────────────────────────────────────────────────────────────────────────────────┐
│                                 MODEL COMPARISON                                 │
├────────────────────────────────────────────────────────────────────────────────┴
    Model A:  deepseek_16B_4xa100_moe_baseline  (20 trials)
    Model B:  deepseek_16B_mod_4xa100  (20 trials)

  Config differences:
    moe_args.mod_capacity_ratio: 0.0 → 0.125
    moe_args.residual_routing: None → False

┌────────────────────────────────┬──────────────────────┬──────────────────────┬────────────┬──────────┐
│ Metric                         │              Model A │              Model B │     % Diff │  Winner  │
├────────────────────────────────┼──────────────────────┼──────────────────────┼────────────┼──────────┤
│   ROUTING EFFICIENCY           │                      │                      │            │          │
├────────────────────────────────┼──────────────────────┼──────────────────────┼────────────┼──────────┤
│     Avg Gini Coefficient       │               0.6654 │               0.6233 │     -6.33% │    B     │
│     Avg Coeff. of Variation    │               1.5713 │               1.4822 │     -5.67% │    B     │
│     Avg Expert Utilization     │               0.9523 │               0.9571 │     +0.51% │    B     │
├────────────────────────────────┼──────────────────────┼──────────────────────┼────────────┼──────────┤
│   INFERENCE PERFORMANCE        │                      │                      │            │          │
├────────────────────────────────┼──────────────────────┼──────────────────────┼────────────┼──────────┤
│     Latency (ms)               │  35,828.82 ±619.8430 │ 96,422.54 ±10,401.11 │   +169.12% │    A     │
│     Throughput (tokens/sec)    │       3.5735 ±0.0609 │       1.3386 ±0.1100 │    -62.54% │    A     │
│     Memory Allocated (GB)      │              17.2886 │              17.2886 │     +0.00% │    A     │
│     Memory Reserved (GB)       │              18.3421 │              18.3419 │     -0.00% │    B     │
├────────────────────────────────┼──────────────────────┼──────────────────────┼────────────┼──────────┤
│   COMPUTATIONAL COST           │                      │                      │            │          │
├────────────────────────────────┼──────────────────────┼──────────────────────┼────────────┼──────────┤
│     TFLOPs                     │        7748241063936 │        7748404641792 │     +0.00% │    A     │
│     Active Params (B)          │              15.7065 │              15.7065 │     +0.00% │    A     │
├────────────────────────────────┼──────────────────────┼──────────────────────┼────────────┼──────────┤
│   MODEL ACCURACY               │                      │                      │            │          │
├────────────────────────────────┼──────────────────────┼──────────────────────┼────────────┼──────────┤
│     hellaswag (acc)            │               0.3280 │               0.3180 │     -3.05% │    A     │
│     hellaswag (acc_norm)       │               0.3200 │               0.3040 │     -5.00% │    A     │
│     arc_easy (acc)             │               0.2960 │               0.2960 │          - │   tie    │
│     arc_easy (acc_norm)        │               0.3440 │               0.3300 │     -4.07% │    A     │
│     mmlu_abstract_algebra (acc │               0.2200 │               0.2200 │          - │   tie    │
│     mmlu_anatomy (acc)         │               0.1852 │               0.1778 │     -4.00% │    A     │
│     mmlu_astronomy (acc)       │               0.1842 │               0.1776 │     -3.57% │    A     │
│     mmlu_college_biology (acc) │               0.2639 │               0.2500 │     -5.26% │    A     │
│     mmlu_college_chemistry (ac │               0.1900 │               0.1900 │          - │   tie    │
│     mmlu_college_computer_scie │               0.2400 │               0.2400 │          - │   tie    │
│     mmlu_college_mathematics ( │               0.2100 │               0.2200 │     +4.76% │    B     │
│     mmlu_college_physics (acc) │               0.2255 │               0.2059 │     -8.70% │    A     │
│     mmlu_computer_security (ac │               0.2800 │               0.2900 │     +3.57% │    B     │
│     mmlu_conceptual_physics (a │               0.2638 │               0.2638 │          - │   tie    │
│     mmlu_electrical_engineerin │               0.2414 │               0.2414 │          - │   tie    │
│     mmlu_elementary_mathematic │               0.2090 │               0.2090 │          - │   tie    │
│     mmlu_high_school_biology ( │               0.1839 │               0.1839 │          - │   tie    │
│     mmlu_high_school_chemistry │               0.1576 │               0.1675 │     +6.25% │    B     │
│     mmlu_high_school_computer_ │               0.2600 │               0.2500 │     -3.85% │    A     │
│     mmlu_high_school_mathemati │               0.2074 │               0.2148 │     +3.57% │    B     │
│     mmlu_high_school_physics ( │               0.1921 │               0.1987 │     +3.45% │    B     │
│     mmlu_high_school_statistic │               0.1528 │               0.1528 │          - │   tie    │
│     mmlu_machine_learning (acc │               0.3214 │               0.3304 │     +2.78% │    B     │
│     mmlu_stem (acc)            │               0.2134 │               0.2138 │     +0.15% │    B     │
├────────────────────────────────┼──────────────────────┼──────────────────────┼────────────┼──────────┤
│   EVALUATION TIME              │                      │                      │            │          │
├────────────────────────────────┼──────────────────────┼──────────────────────┼────────────┼──────────┤
│     Avg Walltime (seconds)     │    1,695.03 ±16.5112 │   3,373.79 ±460.0844 │    +99.04% │    A     │
└────────────────────────────────┴──────────────────────┴──────────────────────┴────────────┴──────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                                 SUMMARY                                  │
├────────────────────────────────────────────────────────────────────────┤
│   A = deepseek_16B_4xa100_mo...                                        │
│   B = deepseek_16B_mod_4xa100                                          │
├────────────────────────────────────────────────────────────────────────┤
│   ROUTING EFFICIENCY:        B (deepseek_16B_mod_4xa100) wins 3-0      │
│   INFERENCE PERFORMANCE:     A (deepseek_16B_4xa100_mo...) wins 3-1    │
│   COMPUTATIONAL COST:        A (deepseek_16B_4xa100_mo...) wins 2-0    │
│   MODEL ACCURACY:            A (deepseek_16B_4xa100_mo...) wins 8-7    │
│   EVALUATION TIME:           A (deepseek_16B_4xa100_mo...) wins 1-0    │
├────────────────────────────────────────────────────────────────────────┤
│   Total:  A: 14 wins  |  B: 11 wins  |  Ties: 9                        │
├────────────────────────────────────────────────────────────────────────┤
│   OVERALL WINNER: deepseek_16B_4xa100_moe_baseline                     │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                        INTERPRETATION GUIDE                        │
├────────────────────────────────────────────────────────────────────┤
│ Winner: A = Model A is better  |  B = Model B is better            │
├────────────────────────────────────────────────────────────────────┤
│ ROUTING EFFICIENCY                                                 │
│   Gini Coefficient:        Lower is better (0=perfect)             │
│   Coeff. of Variation:     Lower is better                         │
│   Expert Utilization:      Higher is better (1.0=100%)             │
├────────────────────────────────────────────────────────────────────┤
│ INFERENCE PERFORMANCE                                              │
│   Latency:                 Lower is better                         │
│   Throughput:              Higher is better                        │
│   Memory:                  Lower is better                         │
├────────────────────────────────────────────────────────────────────┤
│ COMPUTATIONAL COST                                                 │
│   TFLOPs:                  Lower = more efficient                  │
│   Active Params:           Lower = more efficient                  │
├────────────────────────────────────────────────────────────────────┤
│ MODEL ACCURACY (lm_eval)                                           │
│   acc / acc_norm:          Higher is better                        │
└────────────────────────────────────────────────────────────────────┘
```

~~~~~~~~~~~~
## torchtitan_moe fork change 1 end
~~~~~~~~~~~~

## Latest News
- [2025/11] AMD released an [optimized fork](https://github.com/AMD-AGI/torchtitan-amd/tree/main) of `torchtitan` for AMD GPUs.
- [2025/10] We released `torchtitan` [v0.2.0](https://github.com/pytorch/torchtitan/releases).
- [2025/10] SkyPilot now supports `torchtitan`! See the tutorial [here](https://docs.skypilot.co/en/latest/examples/training/torchtitan.html).
- [2025/07] We published [instructions](/torchtitan/models/README.md) on how to add a model to `torchtitan`.
- [2025/04] Our paper was accepted by [ICLR 2025](https://iclr.cc/virtual/2025/poster/29620).
- [2024/12] GPU MODE [lecture](https://www.youtube.com/watch?v=VYWRjcUqW6w) on torchtitan.
- [2024/07] [Presentation](https://pytorch2024.sched.com/event/1fHn3) at PyTorch Conference 2024.


## Overview

`torchtitan` is a PyTorch native platform designed for **rapid experimentation and large-scale training** of generative AI models. As a minimal clean-room implementation of PyTorch native scaling techniques, `torchtitan` provides a flexible foundation for developers to build upon. With `torchtitan` [extension points](docs/extension.md), one can easily create custom extensions tailored to specific needs.

Our mission is to accelerate innovation in the field of generative AI by empowering researchers and developers to explore new modeling architectures and infrastructure techniques.

The Guiding Principles when building `torchtitan`
* Designed to be easy to understand, use and extend for different training purposes.
* Minimal changes to the model code when applying multi-dimensional parallelism.
* Bias towards a clean, minimal codebase while providing basic reusable / swappable components.

`torchtitan` has been showcasing PyTorch's latest distributed training features, via support for pretraining Llama 3.1 LLMs of various sizes.

## Contributing

We look forward to your contributions!

* To accelerate contributions to and innovations around torchtitan, we host an [`experiments`](torchtitan/experiments) folder. New ideas should start there. To contribute, follow the [`experiments guidelines`](torchtitan/experiments/README.md).
* For fixes and contributions to core, follow these [`guidelines`](CONTRIBUTING.md).

## Llama 3.1 training

### Key features available

1. Multi-dimensional composable parallelisms
   - [FSDP2](docs/fsdp.md) with per-parameter sharding
   - [Tensor Parallel](https://pytorch.org/docs/stable/distributed.tensor.parallel.html) (including [async TP](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487))
   - [Pipeline Parallel](https://discuss.pytorch.org/t/distributed-w-torchtitan-training-with-zero-bubble-pipeline-parallelism/214420)
   - [Context Parallel](https://discuss.pytorch.org/t/distributed-w-torchtitan-breaking-barriers-training-long-context-llms-with-1m-sequence-length-in-pytorch-using-context-parallel/215082)
2. [Meta device](https://pytorch.org/docs/stable/meta.html) initialization
3. Selective (layer or operator) and full activation checkpointing
4. [Distributed checkpointing](https://discuss.pytorch.org/t/distributed-w-torchtitan-optimizing-checkpointing-efficiency-with-pytorch-dcp/211250) (including async checkpointing)
   - [Interoperable checkpoints](docs/checkpoint.md) which can be loaded directly into [`torchtune`](https://github.com/pytorch/torchtune) for fine-tuning
5. `torch.compile` support
6. [Float8](https://discuss.pytorch.org/t/distributed-w-torchtitan-enabling-float8-all-gather-in-fsdp2/209323) support ([how-to](docs/float8.md))
7. [MXFP8 training for dense and MoE models](docs/mxfp8.md) on Blackwell GPUs.
7. DDP and HSDP
8. [TorchFT](https://github.com/pytorch/torchft) integration
9. Checkpointable data-loading, with the C4 dataset pre-configured (144M entries) and support for [custom datasets](docs/datasets.md)
10. Gradient accumulation, enabled by giving an additional `--training.global_batch_size` argument in configuration
11. Flexible learning rate scheduler (warmup-stable-decay)
12. Loss, GPU memory, throughput (tokens/sec), TFLOPs, and MFU displayed and logged via [Tensorboard or Weights & Biases](/docs/metrics.md)
13. [Debugging tools](docs/debugging.md) including CPU/GPU profiling, memory profiling, Flight Recorder, etc.
14. All options easily configured via [toml files](torchtitan/models/llama3/train_configs/)
15. [Helper scripts](scripts/) to
    - download tokenizers from Hugging Face
    - convert original Llama 3 checkpoints into the expected DCP format
    - estimate FSDP/HSDP memory usage without materializing the model
    - run distributed inference with Tensor Parallel

We report [performance](benchmarks/llama3_h100_202412_torchtitan.md) on up to 512 GPUs, and verify [loss converging](docs/converging.md) correctness of various techniques.

### Dive into the code

You may want to see how the model is defined or how parallelism techniques are applied. For a guided tour, see these files first:
* [torchtitan/train.py](torchtitan/train.py) - the main training loop and high-level setup code
* [torchtitan/models/llama3/model/model.py](torchtitan/models/llama3/model/model.py) - the Llama 3.1 model definition
* [torchtitan/models/llama3/infra/parallelize.py](torchtitan/models/llama3/infra/parallelize.py) - helpers for applying Data Parallel, Tensor Parallel, activation checkpointing, and `torch.compile` to the model
* [torchtitan/models/llama3/infra/pipeline.py](torchtitan/models/llama3/infra/pipeline.py) - helpers for applying Pipeline Parallel to the model
* [torchtitan/components/checkpoint.py](torchtitan/components/checkpoint.py) - utils for saving/loading distributed checkpoints
* [torchtitan/components/quantization/float8.py](torchtitan/components/quantization/float8.py) - utils for applying Float8 techniques


## Installation

One can directly run the source code, or install `torchtitan` from a nightly build, or a stable release.

### From source

This method requires the nightly build of PyTorch, or the latest PyTorch built [from source](https://github.com/pytorch/pytorch?tab=readme-ov-file#from-source).

```bash
git clone https://github.com/pytorch/torchtitan
cd torchtitan
pip install -r requirements.txt
```

### Nightly builds

This method requires the nightly build of PyTorch. You can replace `cu126` with another version of cuda (e.g. `cu128`) or an AMD GPU (e.g. `rocm6.3`).

```sh
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --force-reinstall
pip install --pre torchtitan --index-url https://download.pytorch.org/whl/nightly/cu126
```

### Stable releases
One can install the latest [stable release](https://github.com/pytorch/torchtitan/releases) of `torchtitan` via `pip` or `conda`.
```sh
pip install torchtitan
```
```sh
conda install conda-forge::torchtitan
```
Note that each stable release pins the nightly versions of `torch` and `torchao`. Please see [release.md](docs/release.md) for more details.

### Downloading a tokenizer

`torchtitan` currently supports training Llama 3.1 (8B, 70B, 405B) out of the box. To get started training these models, we need to download the tokenizer. Follow the instructions on the official [meta-llama](https://huggingface.co/meta-llama/Llama-3.1-8B) repository to ensure you have access to the Llama model weights.

Once you have confirmed access, you can run the following command to download the Llama 3.1 tokenizer to your local machine.

```bash
# Get your HF token from https://huggingface.co/settings/tokens

# Llama 3.1 tokenizer
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token=...
```

### Start a training run
Llama 3 8B model locally on 8 GPUs

```bash
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" ./run_train.sh
```

### Multi-Node Training
For training on ParallelCluster/Slurm type configurations, you can use the `multinode_trainer.slurm` file to submit your sbatch job.

To get started adjust the number of nodes and GPUs
```
#SBATCH --ntasks=2
#SBATCH --nodes=2
```

Then start a run where `nnodes` is your total node count, matching the sbatch node count above.

```
srun torchrun --nnodes 2
```

If your gpu count per node is not 8, adjust `--nproc_per_node` in the torchrun command and `#SBATCH --gpus-per-task` in the SBATCH command section.


## Citation

We provide a detailed look into the parallelisms and optimizations available in `torchtitan`, along with summary advice on when to use various techniques.

[TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training](https://openreview.net/forum?id=SFN6Wm7YBI)
```
@inproceedings{
   liang2025torchtitan,
   title={TorchTitan: One-stop PyTorch native solution for production ready {LLM} pretraining},
   author={Wanchao Liang and Tianyu Liu and Less Wright and Will Constable and Andrew Gu and Chien-Chin Huang and Iris Zhang and Wei Feng and Howard Huang and Junjie Wang and Sanket Purandare and Gokul Nadathur and Stratos Idreos},
   booktitle={The Thirteenth International Conference on Learning Representations},
   year={2025},
   url={https://openreview.net/forum?id=SFN6Wm7YBI}
}
```


## License

Source code is made available under a [BSD 3 license](./LICENSE), however you may have other legal obligations that govern your use of other content linked in this repository, such as the license or terms of service for third-party data and models.

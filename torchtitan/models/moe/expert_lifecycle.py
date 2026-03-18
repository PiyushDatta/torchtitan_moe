# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Expert Lifecycle Manager for MoE: evolutionary expert recycling.

Standard MoE training suffers from routing collapse where deep-layer experts
die (receive zero tokens) while a few dominant experts handle everything.
Load balancing via expert_bias corrections is too slow to counteract the
rich-get-richer feedback loop.

The Expert Lifecycle Manager takes an evolutionary approach: periodically
evaluate expert fitness, prune dead/dying experts, and replace them with
mutated clones of top-performing experts. This recycles wasted parameter
capacity and maintains expert diversity throughout training.

Inspired by:
- Lottery Ticket Hypothesis (Frankle & Carlin, 2019)
- RIGL: Rigging the Lottery (Evci et al., 2020)
- Population Based Training (Jaderberg et al., 2017)
"""

import torch
from torch import nn

from torchtitan.tools.logging import logger


class ExpertLifecycleManager:
    """Evolutionary lifecycle manager for MoE experts.

    Periodically evaluates expert fitness based on accumulated token routing
    patterns, prunes underperforming experts, and replaces them with mutated
    clones of top-performing experts.

    The manager maintains its own fitness accumulators (separate from the
    per-step tokens_per_expert used by expert_bias load balancing) so that
    fitness reflects routing behavior over the full lifecycle interval.

    Args:
        lifecycle_interval: Training steps between lifecycle evaluations.
        prune_ratio: Fraction of experts eligible for pruning each cycle
            (e.g., 0.1 = bottom 10%).
        mutation_scale: Scale of noise added to cloned weights, relative to
            the source weight's standard deviation. Controls how quickly
            clones diverge from their parent.
        prune_threshold: Experts receiving fewer than this fraction of the
            mean tokens are candidates for pruning (e.g., 0.1 = less than
            10% of the average load).
    """

    def __init__(
        self,
        lifecycle_interval: int = 100,
        prune_ratio: float = 0.1,
        mutation_scale: float = 0.01,
        prune_threshold: float = 0.1,
    ):
        self.lifecycle_interval = lifecycle_interval
        self.prune_ratio = prune_ratio
        self.mutation_scale = mutation_scale
        self.prune_threshold = prune_threshold
        self.step_count = 0

        # Accumulated fitness per layer: {layer_key: Tensor(num_experts)}
        # Built lazily on first call to accumulate_fitness.
        self._accumulated_fitness: dict[str, torch.Tensor] = {}

    def accumulate_fitness(
        self,
        model_parts: list[nn.Module],
    ) -> None:
        """Accumulate expert fitness from current step's tokens_per_expert.

        Must be called BEFORE tokens_per_expert is zeroed by the expert_bias
        update hook. This captures each step's routing decisions into the
        lifecycle's own accumulator.
        """
        for model_part in model_parts:
            layers = model_part.get_submodule("layers")
            assert isinstance(layers, nn.ModuleDict)
            for layer_name, transformer_block in layers.items():
                if not transformer_block.moe_enabled:
                    continue
                moe = transformer_block.moe
                key = f"{id(model_part)}_{layer_name}"
                tokens = moe.tokens_per_expert
                if key not in self._accumulated_fitness:
                    self._accumulated_fitness[key] = torch.zeros_like(tokens)
                self._accumulated_fitness[key].add_(tokens)

    def maybe_evolve(
        self,
        model_parts: list[nn.Module],
    ) -> None:
        """Called every training step. Performs lifecycle ops at intervals.

        Call this AFTER accumulate_fitness and AFTER expert_bias update.
        """
        self.step_count += 1
        if self.step_count % self.lifecycle_interval != 0:
            return

        total_pruned = 0
        total_layers = 0

        for model_part in model_parts:
            layers = model_part.get_submodule("layers")
            assert isinstance(layers, nn.ModuleDict)
            for layer_name, transformer_block in layers.items():
                if not transformer_block.moe_enabled:
                    continue
                moe = transformer_block.moe
                key = f"{id(model_part)}_{layer_name}"
                fitness = self._accumulated_fitness.get(key)
                if fitness is None:
                    continue
                n_pruned = self._evolve_layer(layer_name, moe, fitness)
                total_pruned += n_pruned
                total_layers += 1
                # Reset accumulator for next interval
                fitness.zero_()

        if total_pruned > 0:
            logger.info(
                f"[ExpertLifecycle] Step {self.step_count}: "
                f"recycled {total_pruned} experts across {total_layers} MoE layers"
            )

    def _evolve_layer(
        self,
        layer_name: str,
        moe: nn.Module,
        fitness: torch.Tensor,
    ) -> int:
        """Evaluate fitness and perform lifecycle operations on one MoE layer.

        Returns the number of experts that were pruned and replaced.
        """
        num_experts = fitness.shape[0]
        n_prune = max(1, int(num_experts * self.prune_ratio))

        # Rank experts by fitness (ascending = worst first)
        rankings = torch.argsort(fitness)
        bottom = rankings[:n_prune]
        top = rankings[-n_prune:]

        # Only prune experts that are truly dead/dying
        mean_fitness = fitness.mean()
        threshold = mean_fitness * self.prune_threshold

        actually_pruned = 0
        with torch.no_grad():
            for i in range(n_prune):
                bad_idx = bottom[i].item()
                good_idx = top[n_prune - 1 - i].item()

                # Skip if the "bad" expert isn't actually underperforming
                if fitness[bad_idx] >= threshold:
                    continue

                # Skip if the "good" expert is also dead
                if fitness[good_idx] <= threshold:
                    continue

                self._clone_expert_weights(moe.experts, good_idx, bad_idx)
                self._clone_gate_weights(moe.router, good_idx, bad_idx)

                # Reset the expert_bias for the cloned expert to neutral
                if hasattr(moe, "expert_bias") and moe.expert_bias is not None:
                    moe.expert_bias[bad_idx] = 0.0

                actually_pruned += 1
                logger.info(
                    f"  [layer {layer_name}] Recycled expert {bad_idx} "
                    f"(tokens={fitness[bad_idx]:.0f}) <- "
                    f"cloned from expert {good_idx} "
                    f"(tokens={fitness[good_idx]:.0f})"
                )

        return actually_pruned

    def _clone_expert_weights(
        self,
        experts: nn.Module,
        source: int,
        target: int,
    ) -> None:
        """Clone expert FFN weights from source to target with mutation.

        Handles both plain tensors and DTensors (for expert parallelism).
        With EP, only clones if both source and target are on the local shard.
        """
        for param_name in ("w1", "w2", "w3"):
            param = getattr(experts, param_name)

            if isinstance(param, torch.distributed.tensor.DTensor):
                local_param = param.to_local()
                local_num_experts = local_param.shape[0]
                global_num_experts = param.shape[0]

                if local_num_experts < global_num_experts:
                    # Experts are sharded across EP. Map global -> local.
                    source_shard = source // local_num_experts
                    target_shard = target // local_num_experts
                    local_rank = param.device_mesh.get_local_rank(
                        mesh_dim="ep"
                        if "ep" in param.device_mesh.mesh_dim_names
                        else 0
                    )

                    if source_shard != local_rank or target_shard != local_rank:
                        # Source and target on different EP shards -- skip.
                        # The gate weight clone will still redirect tokens
                        # to this expert; its weights will learn via gradients.
                        continue

                    local_source = source % local_num_experts
                    local_target = target % local_num_experts
                else:
                    # Replicated -- use global indices directly
                    local_source = source
                    local_target = target

                src_data = local_param.data[local_source]
                noise = torch.randn_like(src_data) * (
                    self.mutation_scale * src_data.std()
                )
                local_param.data[local_target] = src_data + noise
            else:
                src_data = param.data[source]
                noise = torch.randn_like(src_data) * (
                    self.mutation_scale * src_data.std()
                )
                param.data[target] = src_data + noise

    def _clone_gate_weights(
        self,
        router: nn.Module,
        source: int,
        target: int,
    ) -> None:
        """Clone gate weights from source to target with mutation.

        The mutation ensures the clone receives slightly different tokens
        than its parent, creating selection pressure for specialization.
        Handles both replicated and sharded gate weights (EP).
        """
        gate = router.gate

        if isinstance(gate.weight, torch.distributed.tensor.DTensor):
            local_weight = gate.weight.to_local()
            local_size = local_weight.shape[0]
            global_size = gate.weight.shape[0]

            if local_size < global_size:
                # Gate is sharded (e.g., across EP). Only clone if both
                # source and target are on this shard.
                source_local = source % local_size
                target_local = target % local_size
                source_shard = source // local_size
                target_shard = target // local_size
                local_rank = gate.weight.device_mesh.get_local_rank(
                    mesh_dim="ep"
                    if "ep" in gate.weight.device_mesh.mesh_dim_names
                    else 0
                )
                if source_shard != local_rank or target_shard != local_rank:
                    return
                src_weight = local_weight.data[source_local]
                noise = torch.randn_like(src_weight) * (
                    self.mutation_scale * src_weight.std()
                )
                local_weight.data[target_local] = src_weight + noise
            else:
                # Gate is replicated -- can clone any index directly
                src_weight = local_weight.data[source]
                noise = torch.randn_like(src_weight) * (
                    self.mutation_scale * src_weight.std()
                )
                local_weight.data[target] = src_weight + noise
        else:
            src_weight = gate.weight.data[source]
            noise = torch.randn_like(src_weight) * (
                self.mutation_scale * src_weight.std()
            )
            gate.weight.data[target] = src_weight + noise

        if gate.bias is not None:
            if isinstance(gate.bias, torch.distributed.tensor.DTensor):
                local_bias = gate.bias.to_local()
                local_size = local_bias.shape[0]
                global_size = gate.bias.shape[0]
                if local_size < global_size:
                    source_local = source % local_size
                    target_local = target % local_size
                    source_shard = source // local_size
                    target_shard = target // local_size
                    local_rank = gate.bias.device_mesh.get_local_rank(
                        mesh_dim="ep"
                        if "ep" in gate.bias.device_mesh.mesh_dim_names
                        else 0
                    )
                    if source_shard != local_rank or target_shard != local_rank:
                        return
                    local_bias.data[target_local] = local_bias.data[source_local]
                else:
                    local_bias.data[target] = local_bias.data[source]
            else:
                gate.bias.data[target] = gate.bias.data[source]

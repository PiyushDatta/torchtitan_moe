# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mixture of Depths (MoD) as an MoE subclass.

MoD selects a subset of tokens via a lightweight router and only processes
those through the MoE layer. Unselected tokens produce zero output; the
residual connection in the TransformerBlock handles the pass-through.

By subclassing MoE, all existing parallelization plans, isinstance checks,
attribute access patterns, and checkpoint FQN paths work unchanged.
"""

import torch
from torch import nn

from .moe import MoE, MoEArgs


class MoDRouter(nn.Module):
    """Lightweight router that selects which tokens to process.

    Args:
        dim (int): Input token dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, 1, bias=False)

    def forward(
        self, x: torch.Tensor, capacity: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Flattened input tokens, shape ``(num_tokens, dim)``.
            capacity: Number of tokens to select.

        Returns:
            scores: Sigmoid scores for all tokens, shape ``(num_tokens,)``.
            top_indices: Indices of selected tokens, shape ``(capacity,)``.
        """
        scores = torch.sigmoid(self.gate(x).squeeze(-1).float())
        _, top_indices = torch.topk(scores, k=capacity, sorted=False)
        return scores, top_indices

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)


class MixtureOfDepths(MoE):
    """MoE with Mixture of Depths token selection.

    Only the top-C tokens (C = capacity_ratio * num_tokens) are processed
    through the MoE. Their outputs are weighted by the MoD router scores
    and scattered back to the full sequence. Unselected positions are zero,
    so the residual add in TransformerBlock acts as a skip.

    Since this subclasses MoE, all attributes (router, experts, shared_experts,
    reorderer, etc.) live directly on this module, keeping FQN paths, parallelize
    plans, and isinstance checks identical to standard MoE.

    Args:
        moe_args: MoE configuration.
        dim: Model dimension.
        hidden_dim: Expert hidden dimension.
        capacity_ratio: Fraction of tokens to process (0, 1].
    """

    def __init__(
        self,
        moe_args: MoEArgs,
        dim: int,
        hidden_dim: int,
        capacity_ratio: float,
    ):
        super().__init__(moe_args, dim=dim, hidden_dim=hidden_dim)
        self.mod_router = MoDRouter(dim)
        self.capacity_ratio = capacity_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, slen, dim = x.shape
        num_tokens = bs * slen
        capacity = max(1, int(num_tokens * self.capacity_ratio))

        x_flat = x.view(-1, dim)
        mod_scores, mod_indices = self.mod_router(x_flat, capacity)
        mod_weights = mod_scores[mod_indices]  # (capacity,)

        # Process selected tokens through the MoE
        selected = x_flat[mod_indices].unsqueeze(0)  # (1, capacity, dim)
        moe_out = super().forward(selected).squeeze(0)  # (capacity, dim)

        # Weight by MoD router scores and scatter back
        weighted = (moe_out.float() * mod_weights.unsqueeze(-1)).to(x.dtype)
        result = torch.zeros(num_tokens, dim, dtype=x.dtype, device=x.device)
        result[mod_indices] = weighted
        return result.view(bs, slen, dim)

    def init_weights(self, init_std: float, buffer_device: torch.device):
        super().init_weights(init_std, buffer_device)
        self.mod_router.init_weights(init_std)

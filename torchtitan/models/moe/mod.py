# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mixture of Depths (MoD) for transformer blocks.

Paper: https://arxiv.org/abs/2404.02258

MoD wraps an entire TransformerBlock and uses a learned router to select
a subset of tokens (determined by capacity_ratio) to process through the
block. Unselected tokens skip the block entirely via the residual connection.

This is fundamentally different from only skipping the MoE/FFN — MoD skips
both attention AND the feed-forward/MoE, saving FLOPs from the most expensive
parts of the model.

The implementation includes an auxiliary MLP that learns to predict the
top-k routing decisions during training. At inference time (autoregressive
sampling), the aux MLP is used instead of top-k routing since future tokens
are not available.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoDRouter(nn.Module):
    """Router that selects which tokens to process through the block.

    Uses a linear projection to compute per-token scores. During training,
    top-k selection determines which tokens are processed. An auxiliary MLP
    learns to predict these decisions for use during causal inference.

    Args:
        dim: Input token dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.weight_predictor = nn.Linear(dim, 1, bias=False)
        self.aux_predictor = nn.Linear(dim, 1, bias=False)

    def forward(
        self, x: torch.Tensor, capacity: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tokens, shape ``(batch_size, seq_len, dim)``.
            capacity: Number of tokens to select per batch element.

        Returns:
            router_weights: Raw router logits for all tokens,
                shape ``(batch_size, seq_len)``.
            selected_mask: Boolean mask of selected tokens,
                shape ``(batch_size, seq_len)``.
            aux_loss: Auxiliary BCE loss for training the aux predictor.
                Zero during eval.
        """
        # Router weights: raw logits (not sigmoid)
        router_weights = self.weight_predictor(x).squeeze(-1)  # (B, S)

        if self.training:
            # Top-k selection (non-causal, uses future info)
            _, topk_indices = torch.topk(router_weights, capacity, dim=-1)

            # Binary targets for aux MLP training
            aux_targets = torch.zeros_like(router_weights)
            aux_targets.scatter_(1, topk_indices, 1.0)

            # Aux MLP learns to predict top-k decisions
            # Stop gradient: aux predictor should not affect main model
            aux_logits = self.aux_predictor(x.detach()).squeeze(-1)
            aux_loss = F.binary_cross_entropy_with_logits(aux_logits, aux_targets)

            selected_mask = aux_targets.bool()
        else:
            # At inference, use aux predictor (causal — no future info needed)
            aux_logits = self.aux_predictor(x.detach()).squeeze(-1)
            selected_mask = (torch.sigmoid(aux_logits) > 0.5)
            aux_loss = torch.tensor(0.0, device=x.device)

            # Ensure at least one token is selected per batch element
            # to avoid empty forward passes
            if not selected_mask.any(dim=-1).all():
                # For any batch element with no selections, select the
                # token with the highest aux score
                for b in range(selected_mask.size(0)):
                    if not selected_mask[b].any():
                        best = aux_logits[b].argmax()
                        selected_mask[b, best] = True

        return router_weights, selected_mask, aux_loss

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.weight_predictor.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.aux_predictor.weight, mean=0.0, std=init_std)


class MoDTransformerBlock(nn.Module):
    """Wrapper that applies Mixture-of-Depths routing to a TransformerBlock.

    Only selected tokens (top-C by router score) are processed through the
    wrapped block. Their outputs are weighted by the router scores and placed
    back into the full sequence. Unselected tokens pass through unchanged
    via the residual connection.

    Args:
        block: The TransformerBlock to wrap.
        dim: Model hidden dimension.
        capacity_ratio: Fraction of tokens to process (0, 1]. Paper default: 0.125.
        aux_loss_coeff: Weight for the auxiliary routing loss. Default: 0.01.
    """

    def __init__(
        self,
        block: nn.Module,
        dim: int,
        capacity_ratio: float = 0.125,
        aux_loss_coeff: float = 0.01,
    ):
        super().__init__()
        if capacity_ratio <= 0 or capacity_ratio > 1:
            raise ValueError(
                f"capacity_ratio must be in (0, 1]. Got: {capacity_ratio}"
            )
        self.block = block
        self.mod_router = MoDRouter(dim)
        self.capacity_ratio = capacity_ratio
        self.aux_loss_coeff = aux_loss_coeff
        self.aux_loss = torch.tensor(0.0)

    @property
    def moe_enabled(self):
        """Delegate to wrapped block for compatibility with parallelize/optimizer."""
        return self.block.moe_enabled

    @property
    def moe(self):
        """Delegate to wrapped block for compatibility with parallelize/optimizer."""
        return self.block.moe

    @property
    def weight_init_std(self):
        """Delegate to wrapped block."""
        return self.block.weight_init_std

    @property
    def attention(self):
        """Delegate to wrapped block for parallelize compatibility."""
        return self.block.attention

    @property
    def layer_id(self):
        """Delegate to wrapped block."""
        return self.block.layer_id

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with MoD token selection.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis: Precomputed RoPE frequencies.
            attention_masks: Attention masks (BlockMask or None).
            positions: Position indices for RoPE. Defaults to None.

        Returns:
            Output tensor of shape (batch_size, seq_len, dim).
        """
        B, S, D = x.shape
        capacity = max(1, int(S * self.capacity_ratio))

        # Route tokens
        router_weights, selected_mask, self.aux_loss = self.mod_router(x, capacity)

        # Start with identity (unselected tokens pass through)
        output = x.clone()

        # Process each batch element independently since selections differ
        for b in range(B):
            mask_b = selected_mask[b]  # (S,)
            if not mask_b.any():
                continue

            # Gather selected tokens
            selected_indices = mask_b.nonzero(as_tuple=True)[0]  # (num_selected,)
            selected_tokens = x[b, selected_indices].unsqueeze(0)  # (1, num_selected, D)

            # Build position indices for selected tokens (for correct RoPE)
            if positions is not None:
                selected_positions = positions[b, selected_indices].unsqueeze(0)
            else:
                selected_positions = selected_indices.unsqueeze(0)  # (1, num_selected)

            # Build causal attention mask for selected tokens
            # Token at original position i can attend to token at original position j
            # only if i >= j (causal constraint)
            selected_attn_mask = self._build_causal_mask(
                selected_indices, x.device, x.dtype
            )

            # Run the full transformer block on selected tokens only
            block_out = self.block(
                selected_tokens,
                freqs_cis,
                selected_attn_mask,
                selected_positions,
            )  # (1, num_selected, D)

            # Weight the transform (block output minus input) by router scores,
            # preserving the residual: output = x + sigmoid(w) * (block(x) - x)
            # Paper defines r_i = sigmoid(w · x_i), bounding weights to (0, 1)
            selected_router_weights = torch.sigmoid(router_weights[b, selected_indices])  # (num_selected,)
            transform = block_out.squeeze(0) - selected_tokens.squeeze(0)
            weighted_out = selected_tokens.squeeze(0) + transform * selected_router_weights.unsqueeze(-1)

            # Scatter back
            output[b, selected_indices] = weighted_out.to(x.dtype)

        return output

    @staticmethod
    def _build_causal_mask(
        selected_indices: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build a causal attention mask for the selected token positions.

        Args:
            selected_indices: Original position indices of selected tokens.
            device: Device to create mask on.
            dtype: Dtype for the mask.

        Returns:
            Causal mask of shape (1, 1, num_selected, num_selected).
        """
        n = selected_indices.size(0)
        # positions[i] >= positions[j] means token i can attend to token j
        pos = selected_indices.unsqueeze(0).float()  # (1, n)
        # (n, n): mask[i,j] = True if pos[i] >= pos[j] (query i attends to key j)
        causal = pos.T >= pos  # (n, n), True where attention is allowed
        # Convert to additive mask: 0 for allowed, -inf for blocked
        mask = torch.where(causal, 0.0, float("-inf"))
        return mask.unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)

    def init_weights(self, buffer_device: torch.device):
        """Initialize weights for the wrapped block and MoD router."""
        self.block.init_weights(buffer_device=buffer_device)
        init_std = self.block.weight_init_std
        self.mod_router.init_weights(init_std)

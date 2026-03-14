# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Residual-Routed MoE: routing based on attention residuals.

Standard MoE routes tokens based on the full hidden state h_l, which
converges across tokens in deep layers as attention repeatedly averages
representations. This convergence causes the router to see increasingly
similar inputs, leading to routing collapse (a small subset of experts
receives most tokens while others starve).

Residual routing addresses this by routing on the *attention residual*
(h_l - h_{l-1}), i.e., what the attention layer contributed rather than
what has accumulated. This signal stays diverse across tokens at all
depths because different tokens need different refinements, even when
their accumulated representations have converged.

The experts still process the full hidden state — only the routing
decision uses the residual signal.
"""

import torch
from torch import nn

from .moe import MoE, MoEArgs


class ResidualRoutedMoE(MoE):
    """MoE with residual-based routing to combat routing collapse in deep layers.

    When ``routing_input`` is provided to :meth:`forward`, the router scores
    experts using that tensor (typically the attention layer's output) instead
    of the MoE's own input.  Expert computation still operates on the full
    hidden state, so model capacity is unchanged — only the routing decision
    is redirected to a more diverse signal.

    If ``routing_input`` is ``None``, behaviour is identical to :class:`MoE`.
    """

    def __init__(self, moe_args: MoEArgs, dim: int, hidden_dim: int):
        super().__init__(moe_args, dim=dim, hidden_dim=hidden_dim)
        # The routing input (attention residual) comes from a different
        # distribution than the ffn_norm'd hidden state the gate was designed
        # for.  A dedicated RMSNorm keeps the gate's input well-scaled.
        self.routing_norm = nn.RMSNorm(dim, eps=1e-5)

    def forward(
        self,
        x: torch.Tensor,
        routing_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape ``(bs, slen, dim)``.  Used for expert
                computation (and for routing when *routing_input* is None).
            routing_input: Optional separate tensor for the router, shape
                ``(bs, slen, dim)``.  Typically the attention residual.

        Returns:
            Output tensor, shape ``(bs, slen, dim)``.
        """
        bs, slen, dim = x.shape
        x_flat = x.view(-1, dim)

        # --- Routing: use the residual signal when available ---------------
        if routing_input is not None:
            route_flat = self.routing_norm(routing_input).view(-1, dim)
        else:
            route_flat = x_flat

        (
            top_scores,
            selected_experts_indices,
            num_tokens_per_expert,
        ) = self.router(route_flat, self.expert_bias)

        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)

        # --- From here on, identical to MoE.forward() ----------------------
        (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        ) = self.reorderer(top_scores, selected_experts_indices)

        # Experts process the full hidden state, not the routing signal
        routed_input = x_flat[token_indices_experts_sorted // self.router.top_k]

        if self.score_before_experts:
            routed_input = (
                routed_input.to(torch.float32)
                * top_scores_experts_sorted.reshape(-1, 1)
            ).to(x.dtype)

        routed_output = self.experts(routed_input, num_tokens_per_expert)

        # Shared expert
        out = self.shared_experts(x_flat) if self.shared_experts is not None else None

        # Unsort routed outputs
        routed_output_unsorted = torch.zeros(
            (bs * slen * self.router.top_k, dim),
            dtype=routed_output.dtype,
            device=routed_output.device,
        )
        routed_output_unsorted[token_indices_experts_sorted] = routed_output
        routed_output_unsorted = routed_output_unsorted.reshape(
            -1, self.router.top_k, dim
        )
        if not self.score_before_experts:
            out_experts = (
                torch.bmm(
                    top_scores.reshape(-1, 1, self.router.top_k),
                    routed_output_unsorted.float(),
                )
                .to(x.dtype)
                .squeeze(1)
            )
        else:
            out_experts = routed_output_unsorted.sum(dim=1)

        if out is None:
            return out_experts.reshape(bs, slen, dim)
        return (out + out_experts).reshape(bs, slen, dim)

    def init_weights(self, init_std: float, buffer_device: torch.device):
        super().init_weights(init_std, buffer_device)
        self.routing_norm.reset_parameters()

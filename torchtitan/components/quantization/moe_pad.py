# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FakeTensor adapter for torchao's ``permute_and_pad``."""

from __future__ import annotations

import torch
from torchao.prototype.moe_training.ep.permute import permute_and_pad

from torchtitan.components.quantization.scaled_grouped_mm_meta import (
    ep_unbacked_pad_enabled,
    install_scaled_grouped_mm_meta,
)


def _in_fake_tensor_mode() -> bool:
    return (
        torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE) is not None
    )


def _round_up_dim(dim, alignment: int, num_local_experts: int):
    return ((dim + num_local_experts * alignment + alignment - 1) // alignment) * alignment


def _mint_unbacked_padded_size(alignment: int, num_local_experts: int):
    """Create an unbacked symbol for the padded token dimension."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    fake_mode = torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE)
    assert isinstance(fake_mode, FakeTensorMode)
    shape_env = fake_mode.shape_env
    assert shape_env is not None
    del alignment, num_local_experts
    return shape_env.create_unbacked_symint()


@torch.library.custom_op("torchtitan::permute_and_pad_unbacked", mutates_args=())
def _permute_and_pad_unbacked(
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    ep_degree: int,
    num_local_experts: int,
    alignment: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_shape, x_out, indices, num_padded, offsets = permute_and_pad(
        x,
        num_tokens_per_expert,
        ep_degree,
        num_local_experts,
        alignment,
    )
    # Meta proxy carries ``input_shape`` via ``.shape`` without allocating storage.
    input_shape_proxy = x.new_empty(input_shape, device="meta")
    return input_shape_proxy, x_out, indices, num_padded, offsets


@_permute_and_pad_unbacked.register_fake
def _(
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    ep_degree: int,
    num_local_experts: int,
    alignment: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del ep_degree
    if ep_unbacked_pad_enabled():
        install_scaled_grouped_mm_meta()
        padded_m = _mint_unbacked_padded_size(alignment, num_local_experts)
    else:
        padded_m = _round_up_dim(x.shape[0], alignment, num_local_experts)
    input_shape_proxy = x.new_empty((x.shape[0] + 1, x.shape[-1]), device="meta")
    x_out = x.new_empty((padded_m, x.shape[-1]))
    indices = torch.empty(
        (padded_m,),
        device=num_tokens_per_expert.device,
        dtype=torch.int32,
    )
    num_padded = torch.empty(
        (num_local_experts,),
        device=num_tokens_per_expert.device,
        dtype=torch.int32,
    )
    offsets = torch.empty(
        (num_local_experts,),
        device=num_tokens_per_expert.device,
        dtype=torch.int32,
    )
    return input_shape_proxy, x_out, indices, num_padded, offsets


def _permute_and_pad_unbacked_setup_context(ctx, inputs, output) -> None:
    x = inputs[0]
    _input_shape_proxy, _x_out, indices, _num_padded, _offsets = output
    ctx.save_for_backward(indices)
    # Match permute_and_pad: one sentinel padding row is appended before indexing.
    ctx.input_nrows = x.shape[0] + 1
    ctx.input_ncols = x.shape[-1]


def _permute_and_pad_unbacked_backward(
    ctx,
    grad_input_shape_proxy,
    grad_x_out,
    grad_indices,
    grad_num_padded,
    grad_offsets,
):
    del grad_input_shape_proxy, grad_indices, grad_num_padded, grad_offsets
    (indices,) = ctx.saved_tensors
    if grad_x_out is None:
        return None, None, None, None, None
    # Inverse of x_out = x_padded[indices]: scatter grads back, then drop sentinel.
    grad_input_padded = grad_x_out.new_zeros((ctx.input_nrows, ctx.input_ncols))
    grad_input_padded[indices, :] = grad_x_out
    return grad_input_padded[:-1], None, None, None, None


_permute_and_pad_unbacked.register_autograd(
    _permute_and_pad_unbacked_backward,
    setup_context=_permute_and_pad_unbacked_setup_context,
)


def permute_and_pad_unbacked(
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    ep_degree: int,
    num_local_experts: int,
    alignment: int,
):
    """Same contract as torchao ``permute_and_pad``; custom-op path for FakeTensor/make_fx."""
    # Real tensors must not go through the custom op: torchao ``permute_and_pad``
    # already backprops through indexing.
    if not _in_fake_tensor_mode():
        return permute_and_pad(
            x,
            num_tokens_per_expert,
            ep_degree,
            num_local_experts,
            alignment,
        )

    if ep_unbacked_pad_enabled():
        install_scaled_grouped_mm_meta()

    input_shape_proxy, x_out, indices, num_padded, offsets = _permute_and_pad_unbacked(
        x,
        num_tokens_per_expert,
        ep_degree,
        num_local_experts,
        alignment,
    )
    return input_shape_proxy.shape, x_out, indices, num_padded, offsets

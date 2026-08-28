# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Work around symbolic layout checks in grouped-mm FakeTensor metadata."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

logger = logging.getLogger()

ENV_FLAG = "TORCHTITAN_FP8_EP_UNBACKED_PAD"

_installed = False
_orig_meta_grouped_mm_common = None


def ep_unbacked_pad_enabled() -> bool:
    return os.environ.get(ENV_FLAG, "") == "1"


def _is_data_dependent_guard(exc: BaseException) -> bool:
    from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode

    if isinstance(exc, GuardOnDataDependentSymNode):
        return True
    # FakeTensor sometimes wraps the guard failure.
    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, GuardOnDataDependentSymNode):
        return True
    msg = str(exc)
    return "GuardOnDataDependentSymNode" in msg or "data-dependent" in msg


def _soft_meta_grouped_mm_common(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scale_a: torch.Tensor | None,
    scale_b: torch.Tensor | None,
    offs: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    scale_result: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    use_fast_accum: bool = False,
) -> Any:
    assert _orig_meta_grouped_mm_common is not None
    try:
        return _orig_meta_grouped_mm_common(
            mat_a,
            mat_b,
            scale_a,
            scale_b,
            offs=offs,
            bias=bias,
            scale_result=scale_result,
            out_dtype=out_dtype,
            use_fast_accum=use_fast_accum,
        )
    except Exception as exc:
        if not _is_data_dependent_guard(exc):
            raise
        from torch._meta_registrations import _create_grouped_mm_output_tensor

        scaled = scale_a is not None and scale_b is not None
        if scaled:
            out_dtype = out_dtype or torch.bfloat16
        else:
            out_dtype = out_dtype or mat_a.dtype
        logger.warning(
            "TORCHTITAN_FP8_EP_UNBACKED_PAD=1: skipped data-dependent "
            "aten::_scaled_grouped_mm / grouped_mm meta layout guards "
            "(%s). Falling back to shape-only meta output.",
            type(exc).__name__,
        )
        try:
            return _create_grouped_mm_output_tensor(mat_a, mat_b, offs, out_dtype)
        except Exception as fallback_exc:
            raise fallback_exc from exc


def install_soft_scaled_grouped_mm_meta(*, force: bool = False) -> bool:
    """Install the soft meta wrapper. Returns True if newly installed."""
    global _installed, _orig_meta_grouped_mm_common
    if _installed:
        return False
    if not force and not ep_unbacked_pad_enabled():
        return False

    import torch._meta_registrations as meta_reg

    _orig_meta_grouped_mm_common = meta_reg._meta_grouped_mm_common
    meta_reg._meta_grouped_mm_common = _soft_meta_grouped_mm_common
    _installed = True
    logger.warning(
        "Installed soft aten::_scaled_grouped_mm meta layout fallback "
        "(TORCHTITAN_FP8_EP_UNBACKED_PAD). Temporary GraphTrainer hack."
    )
    return True

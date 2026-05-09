# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Standalone wrapper around the RISE-trained progress value model.

The trained checkpoint at
``RISE/policy_and_value/policy_offline_and_value/checkpoints/value_release_libero_spatial/...``
is a paligemma-backed PI0Pytorch with `with_value_head=True`. We load it as a
side computation off the rollout actor: given the same `extracted_obs` the
rollout already builds for the policy, return scalar V(s) per env.

The actual `openpi_value` package is loaded lazily from the in-tree RISE source
checkout so the rest of rlinf can still import this module without RISE deps.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch

# Add RISE source to sys.path lazily on first construction. Adjust if you ever
# move the RISE clone outside the project root.
_RISE_SRC_CANDIDATES = [
    Path(__file__).resolve().parents[3] / "RISE" / "policy_and_value" / "policy_offline_and_value" / "src",
]


def _ensure_openpi_value_on_path() -> None:
    if any("openpi_value" in m for m in sys.modules):
        return
    for p in _RISE_SRC_CANDIDATES:
        if (p / "openpi_value").is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
            return
    raise RuntimeError(
        "openpi_value source not found. Expected RISE checkout under "
        f"{_RISE_SRC_CANDIDATES[0]}. Set RLINF_RISE_SRC env var to override."
    )


class RiseValueModel:
    """Wraps PI0Pytorch with `with_value_head=True` for inference-only V(s) scoring.

    Construction loads the checkpoint and moves the model to bfloat16 on the
    given device. `forward(extracted_obs)` returns a `[B]` float tensor of V(s)
    values clipped to [0, 1] (sigmoid head, exist_negative_progress=False).
    """

    def __init__(
        self,
        ckpt_path: str | os.PathLike,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        _ensure_openpi_value_on_path()
        from openpi_value.models.pi0_config import Pi0Config_Custom  # noqa: E402
        from openpi_value.models_pytorch.pi0_pytorch import PI0Pytorch  # noqa: E402

        # Mirrors the training config "value_release_libero_spatial" in
        # RISE/.../src/openpi_value/training/config.py.
        config = Pi0Config_Custom(
            pi05=True,
            with_value_head=True,
            loss_value_weight=1.0,
            loss_action_weight=0.0,
            loss_value_use_bce=False,
            p_mask_ego_state=1.0,
            discrete_state_input=False,
            apply_blur_visual_aug=True,
            p_with_progress_loss=1.0,
            exist_negative_progress=False,
            value_TD_learning=True,
            value_TD_TAU=0.01,
            value_gamma=0.995,
            value_terminal_window=10,
            value_failure_reward=-0.6,
        )

        model = PI0Pytorch(config)
        model.eval()

        import safetensors.torch as _sft
        ckpt_path = str(ckpt_path)
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, "model.safetensors")
        # strict=False: action-head weights are random-init at value-only training
        # time; they exist on disk but we do not need them for sample_values.
        _sft.load_model(model, ckpt_path, strict=False)

        self.device = torch.device(device)
        self.dtype = dtype
        self.model = model.to(device=self.device, dtype=self.dtype)
        self.config = config

    @torch.no_grad()
    def forward(self, extracted_obs: dict[str, Any]) -> torch.Tensor:
        """Score a batch of observations.

        Args:
            extracted_obs: rollout-side dict produced by
                ``OpenPi0ForRLActionPrediction.preprocess_env_obs``. Expected
                keys: ``image``/``observation/image`` (dict of camera->tensor),
                ``state`` ([B, state_dim]), ``tokenized_prompt`` ([B, L]),
                ``tokenized_prompt_mask`` ([B, L]).

        Returns:
            [B] float tensor of V(s) on CPU.
        """
        from openpi_value.models.model import Observation  # noqa: E402

        # Normalize key names: rollout produces ``observation/image`` while
        # Observation.from_dict expects ``image``.
        data: dict[str, Any] = {}
        if "image" in extracted_obs:
            data["image"] = extracted_obs["image"]
        elif "observation/image" in extracted_obs:
            data["image"] = extracted_obs["observation/image"]
        else:
            raise KeyError("extracted_obs missing 'image' / 'observation/image'")

        for k_in, k_out in (
            ("state", "state"),
            ("observation/state", "state"),
            ("tokenized_prompt", "tokenized_prompt"),
            ("tokenized_prompt_mask", "tokenized_prompt_mask"),
        ):
            if k_in in extracted_obs and k_out not in data:
                data[k_out] = extracted_obs[k_in]

        # Move tensors to value-model device/dtype where appropriate.
        def _to_device(t: torch.Tensor, target_dtype: torch.dtype | None) -> torch.Tensor:
            t = t.to(self.device)
            if target_dtype is not None and t.dtype != target_dtype:
                t = t.to(dtype=target_dtype)
            return t

        if isinstance(data["image"], dict):
            data["image"] = {k: _to_device(v, self.dtype) for k, v in data["image"].items()}
        else:
            data["image"] = _to_device(data["image"], self.dtype)

        if "state" in data:
            data["state"] = _to_device(data["state"], self.dtype)
        for k in ("tokenized_prompt", "tokenized_prompt_mask"):
            if k in data:
                data[k] = data[k].to(self.device)

        # Observation.from_dict requires data["image_mask"] (a dict of [B] bool
        # tensors keyed by camera). Rollout-side extracted_obs doesn't carry it,
        # so default to "all images valid" when missing.
        if "image_mask" not in data and isinstance(data["image"], dict):
            any_img = next(iter(data["image"].values()))
            B = any_img.shape[0]
            data["image_mask"] = {
                cam: torch.ones(B, dtype=torch.bool, device=self.device)
                for cam in data["image"]
            }

        obs = Observation.from_dict(data)
        # sample_noise() inside sample_values hardcodes dtype=float32, but the
        # model weights are bfloat16. autocast bridges the matmul boundary.
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            v = self.model.sample_values(device=self.device, observation=obs)  # [B, 1]
        return v.detach().to(dtype=torch.float32, device="cpu").reshape(-1)

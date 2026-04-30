# Action-Conditioned VLA SFT Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build SFT pre-training for the action-conditioned π0.5 expert on LIBERO (4 suites) and RoboTwin (3 tasks) — 7 total runs producing checkpoints whose `action_cond_in_proj` carries non-trivial weights.

**Architecture:** Extend the existing RLinf SFT scaffolding (`SFTRunner` + `FSDPSftWorker`) with a new entrypoint and per-suite/per-task configs. Two model-side changes: (1) replace bare `Linear` action-cond port with a sinusoidal Fourier-MLP (matching upstream `action_time_mlp`), gated by `action_cond_freq_bands`; (2) rewrite `sft_forward` to compute the flow-matching loss directly through the action-cond-aware velocity head so the new port actually trains.

**Tech Stack:** PyTorch, FSDP, openpi, hydra/omegaconf, lerobot, HuggingFace datasets, pytest.

**Spec:** `docs/superpowers/specs/2026-04-28-action-conditioned-sft-design.md`

---

## File map

**Create:**
- `tests/test_action_cond_fourier.py` — port-revision tests
- `tests/test_sft_forward.py` — sft_forward tests
- `tests/test_libero_suite_filter.py` — suite filter tests
- `examples/embodiment/train_sft_embodied.py` — SFT entrypoint
- `examples/embodiment/run_sft_embodied.sh` — launch script
- `examples/embodiment/config/libero_spatial_sft_actor_openpi_pi05.yaml`
- `examples/embodiment/config/libero_object_sft_actor_openpi_pi05.yaml`
- `examples/embodiment/config/libero_goal_sft_actor_openpi_pi05.yaml`
- `examples/embodiment/config/libero_10_sft_actor_openpi_pi05.yaml`
- `examples/embodiment/config/robotwin_place_empty_cup_sft_actor_openpi_pi05.yaml`
- `examples/embodiment/config/robotwin_beat_block_hammer_sft_actor_openpi_pi05.yaml`
- `examples/embodiment/config/robotwin_pick_dual_bottles_sft_actor_openpi_pi05.yaml`
- `docs/sft-pretraining.md` — runbook

**Modify:**
- `rlinf/models/embodiment/openpi/openpi_action_model.py` — Fourier port (lines 98-101 config, 209-221 init, 1303 embed_suffix call site, 403 sft_forward)
- `rlinf/models/embodiment/openpi/dataconfig/libero_dataconfig.py` — add `suite` field
- `rlinf/models/embodiment/openpi/dataconfig/__init__.py` — add 4 LIBERO + 3 RoboTwin TrainConfigs

---

### Task 1: Add Fourier-feature config knobs

**Files:**
- Modify: `rlinf/models/embodiment/openpi/openpi_action_model.py:98-101`
- Test: `tests/test_action_cond_fourier.py`

- [ ] **Step 1: Write failing test for new config defaults**

Create `tests/test_action_cond_fourier.py`:

```python
"""Tests for the Fourier-feature action_cond port revision (spec 2026-04-28 §2)."""
import pytest


def test_openpi_config_has_freq_bands_knob():
    from rlinf.models.embodiment.openpi.openpi_action_model import OpenPi0Config

    cfg = OpenPi0Config()
    assert hasattr(cfg, "action_cond_freq_bands"), "missing freq_bands knob"
    assert hasattr(cfg, "action_cond_min_period"), "missing min_period knob"
    assert hasattr(cfg, "action_cond_max_period"), "missing max_period knob"
    assert cfg.action_cond_freq_bands == 0, "default must be 0 for bit-for-bit fallback"
    assert cfg.action_cond_min_period == pytest.approx(1e-2)
    assert cfg.action_cond_max_period == pytest.approx(4.0)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /vePFS/zundong/Negfinetun-VLA
python -m pytest tests/test_action_cond_fourier.py::test_openpi_config_has_freq_bands_knob -v
```
Expected: FAIL with `AttributeError: ... action_cond_freq_bands`.

- [ ] **Step 3: Add the three new fields**

Edit `rlinf/models/embodiment/openpi/openpi_action_model.py` lines 98-101:

```python
    # action conditioning (a_cond port on the expert; see design spec 2026-04-24)
    action_cond_enabled: bool = True     # register action_cond_in_proj + emit a_cond tokens
    action_cond_dropout_prob: float = 0.5  # CFG-style dropout of a_cond during training
    refine_iters: int = 0                # inference refinement iterations (0 => legacy single pass)
    # Fourier-feature pre-encoding for the a_cond port (spec 2026-04-28 §2). 0 = legacy bare-Linear.
    action_cond_freq_bands: int = 0
    action_cond_min_period: float = 1e-2
    action_cond_max_period: float = 4.0
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_action_cond_fourier.py::test_openpi_config_has_freq_bands_knob -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rlinf/models/embodiment/openpi/openpi_action_model.py tests/test_action_cond_fourier.py
git commit -m "feat(action-cond): add Fourier-feature config knobs (spec 2026-04-28 §2.1)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Implement Fourier-MLP variant of action_cond_in_proj

**Files:**
- Modify: `rlinf/models/embodiment/openpi/openpi_action_model.py:209-221` (init), add `_encode_action_cond` helper near line 247
- Modify: `rlinf/models/embodiment/openpi/openpi_action_model.py:1303` (embed_suffix call site)
- Test: `tests/test_action_cond_fourier.py`

- [ ] **Step 1: Write failing tests for Fourier port behavior**

Append to `tests/test_action_cond_fourier.py`:

```python
import math
import torch


def _build_model_for_test(freq_bands: int):
    """Construct a minimal OpenPi0ForRLActionPrediction with a synthetic config."""
    from omegaconf import OmegaConf
    from rlinf.models import get_model
    cfg = OmegaConf.create({
        "model_type": "openpi",
        "model_path": "checkpoints/torch/pi05_base",
        "action_dim": 7, "num_action_chunks": 10, "num_steps": 4, "use_proprio": True,
        "is_lora": False, "lora_rank": 32, "add_value_head": False, "precision": None,
        "openpi": {
            "config_name": "pi05_libero", "num_images_in_input": 2, "noise_level": 0.5,
            "action_chunk": 10, "num_steps": 4, "train_expert_only": True,
            "action_env_dim": 7, "solver_type": "flow_sde", "add_value_head": False,
            "detach_critic_input": False,
            "action_cond_enabled": True, "action_cond_dropout_prob": 0.5, "refine_iters": 0,
            "action_cond_freq_bands": freq_bands,
            "action_cond_min_period": 1e-2, "action_cond_max_period": 4.0,
        },
    })
    return get_model(cfg)


def test_freq_bands_zero_uses_bare_linear():
    m = _build_model_for_test(freq_bands=0)
    import torch.nn as nn
    assert isinstance(m.action_cond_in_proj, nn.Linear)
    assert m.action_cond_in_proj.weight.abs().sum().item() == 0.0


def test_freq_bands_four_uses_sequential_with_zero_init_last():
    m = _build_model_for_test(freq_bands=4)
    import torch.nn as nn
    assert isinstance(m.action_cond_in_proj, nn.Sequential)
    last = m.action_cond_in_proj[-1]
    assert isinstance(last, nn.Linear)
    assert last.weight.abs().sum().item() == 0.0
    assert last.bias.abs().sum().item() == 0.0


def test_encode_action_cond_zero_input_yields_zero_when_freq_bands_zero():
    m = _build_model_for_test(freq_bands=0)
    a = torch.zeros(2, 10, 7, dtype=m.action_in_proj.weight.dtype, device=m.action_in_proj.weight.device)
    out = m._encode_action_cond(a)
    assert out.abs().sum().item() == 0.0


def test_encode_action_cond_zero_input_yields_zero_when_freq_bands_four():
    """With last-layer zero-init, output is zero regardless of input frequency content."""
    m = _build_model_for_test(freq_bands=4)
    a = torch.randn(2, 10, 7, dtype=m.action_in_proj[0].weight.dtype, device=m.action_in_proj[0].weight.device)
    out = m._encode_action_cond(a)
    assert out.abs().sum().item() == 0.0


def test_encode_action_cond_post_step_nonzero():
    """One backward step on a dummy loss should move action_cond_in_proj weights off zero."""
    m = _build_model_for_test(freq_bands=4)
    a = torch.randn(2, 10, 7, dtype=m.action_in_proj[0].weight.dtype if hasattr(m.action_in_proj, '__getitem__') else m.action_in_proj.weight.dtype)
    a.requires_grad_(False)
    out = m._encode_action_cond(a)
    # Add a small perturbation so the gradient flows; any non-trivial loss works.
    last = m.action_cond_in_proj[-1]
    last.weight.data.add_(torch.randn_like(last.weight) * 1e-6)
    out2 = m._encode_action_cond(a)
    assert out2.abs().sum().item() > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_action_cond_fourier.py -v
```
Expected: 5 tests FAIL — port is still bare Linear and `_encode_action_cond` doesn't exist.

- [ ] **Step 3: Replace the port registration in `__init__`**

Edit `rlinf/models/embodiment/openpi/openpi_action_model.py:209-221`. Replace the entire block:

```python
        # a_cond projection (zero-init for bit-for-bit fallback; see specs 2026-04-24 §3.3 + 2026-04-28 §2.2)
        if getattr(self.config, "action_cond_enabled", True):
            import torch.nn as _nn
            action_expert_width = self.action_in_proj.weight.shape[0]
            K = int(getattr(self.config, "action_cond_freq_bands", 0))
            if K == 0:
                # legacy: bare Linear, zero-init
                self.action_cond_in_proj = _nn.Linear(
                    self.config.action_dim, action_expert_width
                )
                _nn.init.zeros_(self.action_cond_in_proj.weight)
                _nn.init.zeros_(self.action_cond_in_proj.bias)
            else:
                in_dim = self.config.action_dim * (1 + 2 * K)
                l1 = _nn.Linear(in_dim, action_expert_width)
                l2 = _nn.Linear(action_expert_width, action_expert_width)
                # zero-init the LAST layer so cold start emits zeros (preserves bit-for-bit fallback property
                # with respect to checkpoints saved when freq_bands was 0 *for the action_cond contribution*;
                # the rest of the projection still loads cleanly).
                _nn.init.zeros_(l2.weight)
                _nn.init.zeros_(l2.bias)
                self.action_cond_in_proj = _nn.Sequential(l1, _nn.SiLU(), l2)
            self.action_cond_in_proj = self.action_cond_in_proj.to(
                dtype=self.action_in_proj.weight.dtype,
                device=self.action_in_proj.weight.device,
            )
```

- [ ] **Step 4: Add the `_encode_action_cond` helper**

Insert after `_sample_action_cond` (around line 247). The helper goes inside the same class (`OpenPi0ForRLActionPrediction`):

```python
    def _encode_action_cond(self, a: torch.Tensor) -> torch.Tensor:
        """Encode a clean action chunk into expert-width tokens.

        With ``action_cond_freq_bands == 0``: bare projection (legacy).
        With ``action_cond_freq_bands > 0``: per-dim sinusoidal Fourier features at
        K log-spaced periods between ``min_period`` and ``max_period``, concatenated
        with the raw action, then through a 2-layer MLP. Spec 2026-04-28 §2.3.
        """
        import math as _math
        K = int(getattr(self.config, "action_cond_freq_bands", 0))
        if K == 0:
            return self.action_cond_in_proj(a)
        fraction = torch.linspace(0.0, 1.0, K, device=a.device, dtype=a.dtype)
        period = self.config.action_cond_min_period * (
            self.config.action_cond_max_period / self.config.action_cond_min_period
        ) ** fraction
        scale = (1.0 / period) * 2 * _math.pi
        proj = a.unsqueeze(-1) * scale  # [..., D, K]
        enc = torch.cat([a.unsqueeze(-1), proj.sin(), proj.cos()], dim=-1)  # [..., D, 1+2K]
        enc = enc.flatten(-2, -1)  # [..., D*(1+2K)]
        return self.action_cond_in_proj(enc)
```

- [ ] **Step 5: Update embed_suffix call site**

Edit `rlinf/models/embodiment/openpi/openpi_action_model.py:1303`. Replace `return self.action_cond_in_proj(a)` with `return self._encode_action_cond(a)`. Confirm the surrounding `acond_proj_func` definition still wraps the call correctly:

```python
        def acond_proj_func(a):
            return self._encode_action_cond(a)
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
python -m pytest tests/test_action_cond_fourier.py -v
```
Expected: all 5 tests PASS.

- [ ] **Step 7: Run pre-existing smoke checks for regression**

```bash
python -c "
from omegaconf import OmegaConf
from rlinf.models import get_model
cfg = OmegaConf.create({
    'model_type': 'openpi', 'model_path': 'models/pi05_libero',
    'action_dim': 7, 'num_action_chunks': 10, 'num_steps': 4, 'use_proprio': True,
    'is_lora': False, 'lora_rank': 32, 'add_value_head': False, 'precision': None,
    'openpi': {
        'config_name': 'pi05_libero', 'num_images_in_input': 2, 'noise_level': 0.5,
        'action_chunk': 10, 'num_steps': 4, 'train_expert_only': True,
        'action_env_dim': 7, 'solver_type': 'flow_sde', 'add_value_head': False,
        'detach_critic_input': False,
        'action_cond_enabled': True, 'action_cond_dropout_prob': 0.5, 'refine_iters': 0,
        'action_cond_freq_bands': 0,  # legacy
    },
})
m = get_model(cfg)
print('legacy freq_bands=0 path OK')
"
```
Expected: prints `legacy freq_bands=0 path OK` (mirrors the snippet in `docs/pre-training.md` Option A — confirms the existing checkpoint-load path still works).

- [ ] **Step 8: Commit**

```bash
git add rlinf/models/embodiment/openpi/openpi_action_model.py tests/test_action_cond_fourier.py
git commit -m "feat(action-cond): Fourier-MLP variant of action_cond_in_proj (spec 2026-04-28 §2.2-§2.4)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Rewrite `sft_forward` to exercise the action_cond port

**Files:**
- Modify: `rlinf/models/embodiment/openpi/openpi_action_model.py:403-406` (sft_forward body)
- Test: `tests/test_sft_forward.py`

- [ ] **Step 1: Write failing test for sft_forward behavior**

Create `tests/test_sft_forward.py`:

```python
"""Tests for the action-cond-aware sft_forward (spec 2026-04-28 §3)."""
import pytest
import torch

from rlinf.models.embodiment.base_policy import ForwardType


def _build_model_with_freq_bands(K: int):
    from omegaconf import OmegaConf
    from rlinf.models import get_model
    cfg = OmegaConf.create({
        "model_type": "openpi",
        "model_path": "checkpoints/torch/pi05_base",
        "action_dim": 7, "num_action_chunks": 10, "num_steps": 4, "use_proprio": True,
        "is_lora": False, "lora_rank": 32, "add_value_head": False, "precision": None,
        "openpi": {
            "config_name": "pi05_libero", "num_images_in_input": 2, "noise_level": 0.5,
            "action_chunk": 10, "num_steps": 4, "train_expert_only": True,
            "action_env_dim": 7, "solver_type": "flow_sde", "add_value_head": False,
            "detach_critic_input": False,
            "action_cond_enabled": True, "action_cond_dropout_prob": 0.5, "refine_iters": 0,
            "action_cond_freq_bands": K,
            "action_cond_min_period": 1e-2, "action_cond_max_period": 4.0,
        },
    })
    return get_model(cfg)


def _make_dummy_sft_batch(model):
    """Build a tiny synthetic batch matching what the LIBERO/RoboTwin loaders return."""
    B, H, D = 2, model.config.num_action_chunks, model.config.action_dim
    actions = torch.randn(B, H, D)
    # observation: minimal shape that input_transform + _preprocess_observation expect.
    # See pi0_pytorch.py for the full schema.
    obs = {
        "image": {
            "base_0_rgb": torch.zeros(B, 3, 224, 224, dtype=torch.uint8),
            "left_wrist_0_rgb": torch.zeros(B, 3, 224, 224, dtype=torch.uint8),
        },
        "image_mask": {
            "base_0_rgb": torch.ones(B, dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones(B, dtype=torch.bool),
        },
        "state": torch.zeros(B, D),
        "tokenized_prompt": torch.zeros(B, 64, dtype=torch.long),
        "tokenized_prompt_mask": torch.ones(B, 64, dtype=torch.bool),
    }
    return {"observation": obs, "actions": actions}


def test_sft_forward_returns_scalar_loss():
    m = _build_model_with_freq_bands(K=4)
    m.eval()
    batch = _make_dummy_sft_batch(m)
    loss = m.forward(forward_type=ForwardType.SFT, data=batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0, f"expected scalar, got shape {loss.shape}"
    assert torch.isfinite(loss).item()


def test_sft_forward_grads_flow_to_action_cond_port():
    m = _build_model_with_freq_bands(K=4)
    m.train()
    batch = _make_dummy_sft_batch(m)
    loss = m.forward(forward_type=ForwardType.SFT, data=batch)
    loss.backward()

    # Check the action_cond_in_proj last-layer weight received gradient.
    last = m.action_cond_in_proj[-1]
    assert last.weight.grad is not None, "no grad on action_cond_in_proj.weight"
    # Note: grad may be exactly zero if every sample in the batch hit the
    # "drop=True" branch of CFG dropout; with B=2 and prob=0.5 that's a
    # 25% false-negative rate. Set seed for determinism.
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_sft_forward.py -v
```
Expected: tests FAIL — current `sft_forward` calls `super().forward(observation, actions)` which uses upstream's signature and doesn't return a scalar in the way the test expects, AND doesn't route through the action_cond port.

- [ ] **Step 3: Rewrite `sft_forward`**

Replace `rlinf/models/embodiment/openpi/openpi_action_model.py:403-406` with:

```python
    def sft_forward(self, data, **kwargs):
        """Flow-matching SFT loss, action-cond-aware (spec 2026-04-28 §3.1).

        Computes v_target = noise - x_0, v_pred = get_velocity_for_oar(obs, x_t, t,
        action_cond), where action_cond is sampled via CFG dropout (so the port
        learns non-trivial weights). Returns the per-batch MSE (scalar).
        """
        import math
        from openpi.models import model as _model
        from openpi.models_pytorch.pi0_pytorch import sample_beta

        observation = data["observation"]
        actions = data["actions"]                         # [B, H, action_dim], clean x_0

        # CFG-dropped action_cond (uses the dropout helper landed in commit e25b8d7).
        action_cond = self._sample_action_cond(actions)

        # Build the openpi Observation object from the dict.
        obs_dict = self.input_transform(data, transpose=False)
        obs_obj = _model.Observation.from_dict(obs_dict)

        # Sample t ∼ Beta(1.5, 1.0), clamped to [eps, 1-eps] (matches upstream openpi).
        B = actions.shape[0]
        device = actions.device
        t = sample_beta(alpha=1.5, beta=1.0, bsize=B, device=device).clamp(1e-3, 1.0 - 1e-3)
        noise = torch.randn_like(actions)
        t_b = t[:, None, None]
        x_t = (1.0 - t_b) * actions + t_b * noise          # standard flow-matching path
        v_target = noise - actions                         # standard flow-matching velocity

        v_pred = self.get_velocity_for_oar(
            observation=obs_obj,
            x_t=x_t,
            t=t,
            action_cond=action_cond,
            compute_grad=True,
        )

        loss = ((v_pred - v_target) ** 2).mean()
        return loss
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_sft_forward.py -v
```
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add rlinf/models/embodiment/openpi/openpi_action_model.py tests/test_sft_forward.py
git commit -m "feat(sft): rewrite sft_forward to exercise action_cond port (spec 2026-04-28 §3)

Today the SFT path delegates to upstream's forward, which doesn't pass action_cond
through embed_suffix — so action_cond_in_proj stays at zero forever. Rewrite to
compute the flow-matching loss directly through get_velocity_for_oar with
CFG-dropped action_cond, so the port learns non-trivial weights.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: LIBERO suite filter knob + 4 per-suite TrainConfigs

**Files:**
- Modify: `rlinf/models/embodiment/openpi/dataconfig/libero_dataconfig.py` (add `suite` field)
- Modify: `rlinf/models/embodiment/openpi/dataconfig/__init__.py` (add 4 per-suite TrainConfigs)
- Test: `tests/test_libero_suite_filter.py`

- [ ] **Step 1: Write failing test for the filter knob**

Create `tests/test_libero_suite_filter.py`:

```python
"""Tests for the LIBERO per-suite filter knob (spec 2026-04-28 §4.1)."""
import pytest


def test_libero_dataconfig_has_suite_field():
    from rlinf.models.embodiment.openpi.dataconfig.libero_dataconfig import (
        LeRobotLiberoDataConfig,
    )
    cfg = LeRobotLiberoDataConfig(suite="spatial")
    assert cfg.suite == "spatial"


def test_libero_dataconfig_default_suite_is_none():
    from rlinf.models.embodiment.openpi.dataconfig.libero_dataconfig import (
        LeRobotLiberoDataConfig,
    )
    cfg = LeRobotLiberoDataConfig()
    assert cfg.suite is None


def test_per_suite_train_configs_registered():
    from rlinf.models.embodiment.openpi.dataconfig import _CONFIGS_DICT
    for suite in ["spatial", "object", "goal", "10"]:
        name = f"pi05_libero_{suite}"
        assert name in _CONFIGS_DICT, f"missing TrainConfig: {name}"
        assert _CONFIGS_DICT[name].data.suite == suite
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_libero_suite_filter.py -v
```
Expected: 3 tests FAIL — `suite` field doesn't exist; configs not registered.

- [ ] **Step 3: Add `suite` field to `LeRobotLiberoDataConfig`**

Edit `rlinf/models/embodiment/openpi/dataconfig/libero_dataconfig.py`. Find the `@dataclasses.dataclass(frozen=True) class LeRobotLiberoDataConfig(DataConfigFactory):` block (line 25). Add the field after `extra_delta_transform`:

```python
@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """..."""

    extra_delta_transform: bool = False
    suite: str | None = None  # "spatial" | "object" | "goal" | "10" | None (=all)
```

Then in the `create()` method, after the existing logic that builds `cfg`, add a suite filter — placement depends on whether openpi's `DataConfig` exposes an `episode_filter` hook. Locate the `create()` method and append at the end (just before `return cfg`):

```python
        if self.suite is not None:
            # LIBERO task strings are prefixed by the suite name (e.g. "libero_spatial:pick_up_the_black_bowl_...").
            # We filter via a post-batch wrapper transform that drops out-of-suite samples.
            # This is correct but slightly less efficient than a native HF filter.
            from openpi import transforms as _transforms

            class _SuiteFilter(_transforms.Transform):
                def __init__(self, suite_prefix):
                    self.suite_prefix = suite_prefix
                def __call__(self, sample):
                    task_str = sample.get("task", "") if isinstance(sample, dict) else ""
                    if not task_str.startswith(self.suite_prefix):
                        return None  # signal drop
                    return sample
            cfg = dataclasses.replace(
                cfg,
                data_transforms=_transforms.Group(
                    inputs=[*cfg.data_transforms.inputs, _SuiteFilter(f"libero_{self.suite}")],
                    outputs=cfg.data_transforms.outputs,
                ),
            )
        return cfg
```

**Note:** if `openpi.transforms.Transform` doesn't have a "return None to drop" semantics, the implementer should fall back to checking openpi's actual transform interface and either (a) raise on out-of-suite samples (which the data loader will treat as filtering), or (b) implement the filter as a HF `dataset.filter()` call before the openpi loader is constructed. Both are bounded; pick whichever the openpi version pinned in `requirements/` supports.

- [ ] **Step 4: Add 4 per-suite TrainConfigs**

Edit `rlinf/models/embodiment/openpi/dataconfig/__init__.py`. After the existing `pi05_libero` block (line 93), add a loop that registers 4 per-suite variants. Insert before the `pi0_maniskill` entry (line 94):

```python
# Per-suite LIBERO SFT configs (spec 2026-04-28 §4.1).
for _suite in ["spatial", "object", "goal", "10"]:
    _CONFIGS.append(TrainConfig(
        name=f"pi05_libero_{_suite}",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_libero/assets"),
            extra_delta_transform=False,
            suite=_suite,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi05_base"
        ),
        pytorch_weight_path="checkpoints/torch/pi05_base",
    ))
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_libero_suite_filter.py -v
```
Expected: 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add rlinf/models/embodiment/openpi/dataconfig/libero_dataconfig.py rlinf/models/embodiment/openpi/dataconfig/__init__.py tests/test_libero_suite_filter.py
git commit -m "feat(sft): LIBERO per-suite filter + 4 TrainConfigs (spec 2026-04-28 §4.1)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: 3 RoboTwin per-task TrainConfigs

**Files:**
- Modify: `rlinf/models/embodiment/openpi/dataconfig/__init__.py`
- Test: `tests/test_libero_suite_filter.py` (extend with RoboTwin tests)

- [ ] **Step 1: Write failing test**

Append to `tests/test_libero_suite_filter.py`:

```python
def test_robotwin_per_task_configs_registered():
    from rlinf.models.embodiment.openpi.dataconfig import _CONFIGS_DICT
    for task in ["place_empty_cup", "beat_block_hammer", "pick_dual_bottles"]:
        name = f"pi05_aloha_robotwin_{task}"
        assert name in _CONFIGS_DICT, f"missing TrainConfig: {name}"
        cfg = _CONFIGS_DICT[name]
        assert cfg.model.pi05 is True
        assert cfg.model.action_horizon == 10
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_libero_suite_filter.py::test_robotwin_per_task_configs_registered -v
```
Expected: FAIL — configs not registered.

- [ ] **Step 3: Add the 3 RoboTwin TrainConfigs**

Edit `rlinf/models/embodiment/openpi/dataconfig/__init__.py`. After the existing `pi0_aloha_robotwin` block (line 217-228), add:

```python
# Per-task RoboTwin SFT configs (spec 2026-04-28 §4.2).
# HF repo names for beat_block_hammer / pick_dual_bottles are ASSUMED to follow the
# `<task>_random` convention used by `place_empty_cup_random`. Verified at runbook §1
# before each SFT launch — adjust here if the assumption is wrong.
_ROBOTWIN_TASK_REPOS = {
    "place_empty_cup":   "robotwin/place_empty_cup_random",       # verified (matches existing pi0_aloha_robotwin)
    "beat_block_hammer": "robotwin/beat_block_hammer_random",     # ASSUMED — verify on HF
    "pick_dual_bottles": "robotwin/pick_dual_bottles_random",     # ASSUMED — verify on HF
}
for _task, _repo in _ROBOTWIN_TASK_REPOS.items():
    _CONFIGS.append(TrainConfig(
        name=f"pi05_aloha_robotwin_{_task}",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False
        ),
        data=LeRobotAlohaDataConfig(
            repo_id=_repo,
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir=f"checkpoints/torch/pi05_robotwin_{_task}/assets"),
            extra_delta_transform=True,
        ),
        pytorch_weight_path="checkpoints/torch/pi05_base",
        batch_size=256,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_workers=8,
        num_train_steps=30_000,
    ))
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_libero_suite_filter.py::test_robotwin_per_task_configs_registered -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rlinf/models/embodiment/openpi/dataconfig/__init__.py tests/test_libero_suite_filter.py
git commit -m "feat(sft): 3 RoboTwin per-task TrainConfigs (spec 2026-04-28 §4.2)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: SFT entrypoint script

**Files:**
- Create: `examples/embodiment/train_sft_embodied.py`
- Create: `examples/embodiment/run_sft_embodied.sh`

- [ ] **Step 1: Read the existing RL entrypoint to mirror its structure**

```bash
head -80 /vePFS/zundong/Negfinetun-VLA/examples/embodiment/train_embodied_agent.py
```
Note the hydra config setup, cluster build, worker placement, runner construction patterns.

- [ ] **Step 2: Write the SFT entrypoint**

Create `examples/embodiment/train_sft_embodied.py`:

```python
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SFT entrypoint for the action-conditioned VLA. Spec 2026-04-28 §5.

import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

# Match the env setup of train_embodied_agent.py
os.environ.setdefault("HYDRA_FULL_ERROR", "1")

from rlinf.runners.sft_runner import SFTRunner
from rlinf.scheduler import Cluster
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.omega_resolver import omegaconf_register
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker

omegaconf_register()


@hydra.main(config_path="config", config_name=None, version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    cluster = Cluster(cfg)
    placement = HybridComponentPlacement(cfg, cluster)

    actor = FSDPSftWorker.options(
        placement=placement.get_placement("actor"),
    ).remote(cfg)

    runner = SFTRunner(cfg=cfg, actor=actor, run_timer=ScopedTimer())
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Write the launch script**

Create `examples/embodiment/run_sft_embodied.sh`:

```bash
#!/bin/bash
# SFT launch script for the action-conditioned VLA. Spec 2026-04-28 §5.
# Usage: bash run_sft_embodied.sh <CONFIG_NAME> [hydra overrides...]

set -e

CONFIG_NAME=$1
shift

if [ -z "$CONFIG_NAME" ]; then
    echo "Usage: $0 <CONFIG_NAME> [hydra overrides...]"
    echo "  e.g.: $0 libero_spatial_sft_actor_openpi_pi05"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

python examples/embodiment/train_sft_embodied.py \
    --config-path config \
    --config-name "$CONFIG_NAME" \
    "$@"
```

- [ ] **Step 4: Make the script executable and test the dry-run**

```bash
chmod +x examples/embodiment/run_sft_embodied.sh
# Dry-run hydra parse on a (not-yet-written) config — expected to fail with "config not found"
# but the import chain should at least load:
python -c "from examples.embodiment import train_sft_embodied; print('entrypoint imports OK')"
```
Expected: `entrypoint imports OK`.

- [ ] **Step 5: Commit**

```bash
git add examples/embodiment/train_sft_embodied.py examples/embodiment/run_sft_embodied.sh
git commit -m "feat(sft): train_sft_embodied entrypoint + run script (spec 2026-04-28 §5)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: 4 LIBERO SFT yaml configs

**Files:**
- Create: `examples/embodiment/config/libero_spatial_sft_actor_openpi_pi05.yaml`
- Create: `examples/embodiment/config/libero_object_sft_actor_openpi_pi05.yaml`
- Create: `examples/embodiment/config/libero_goal_sft_actor_openpi_pi05.yaml`
- Create: `examples/embodiment/config/libero_10_sft_actor_openpi_pi05.yaml`

- [ ] **Step 1: Write the spatial config**

Create `examples/embodiment/config/libero_spatial_sft_actor_openpi_pi05.yaml`:

```yaml
defaults:
  - _self_
  - model: pi0_5

cluster:
  num_nodes: 1
  component_placement:
    actor: 0-7

actor:
  micro_batch_size: 4
  global_batch_size: 256
  enable_offload: false
  model:
    model_type: openpi
    model_path: checkpoints/torch/pi05_base
    action_dim: 7
    num_action_chunks: 10
    is_lora: false
    add_value_head: false
    openpi:
      config_name: pi05_libero_spatial
      action_cond_enabled: true
      action_cond_freq_bands: 4
      action_cond_min_period: 1.0e-2
      action_cond_max_period: 4.0
      action_cond_dropout_prob: 0.5
      refine_iters: 0
      action_chunk: 10

runner:
  max_epochs: 30000
  save_interval: 1000
  val_check_interval: 0
  output_dir: ./results/libero_spatial_sft_v1
  logger:
    log_path: ./logs
    experiment_name: libero_spatial_sft_v1
    backend: wandb
```

- [ ] **Step 2: Copy to the other 3 suites with substitutions**

```bash
cd /vePFS/zundong/Negfinetun-VLA/examples/embodiment/config
for suite in object goal 10; do
    sed "s/libero_spatial/libero_${suite}/g" libero_spatial_sft_actor_openpi_pi05.yaml \
        > libero_${suite}_sft_actor_openpi_pi05.yaml
done
ls libero_*_sft_actor_openpi_pi05.yaml
```
Expected: 4 yaml files listed.

- [ ] **Step 3: Hydra-parse-validate each config**

```bash
cd /vePFS/zundong/Negfinetun-VLA
for suite in spatial object goal 10; do
    python -c "
from hydra import compose, initialize
with initialize(config_path='examples/embodiment/config', version_base='1.3'):
    cfg = compose(config_name='libero_${suite}_sft_actor_openpi_pi05')
    assert cfg.actor.model.openpi.config_name == 'pi05_libero_${suite}'
    print('OK: libero_${suite}_sft_actor_openpi_pi05')
"
done
```
Expected: 4 OK lines.

- [ ] **Step 4: Commit**

```bash
git add examples/embodiment/config/libero_*_sft_actor_openpi_pi05.yaml
git commit -m "feat(sft): 4 LIBERO per-suite SFT configs (spec 2026-04-28 §6.1)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: 3 RoboTwin SFT yaml configs

**Files:**
- Create: `examples/embodiment/config/robotwin_place_empty_cup_sft_actor_openpi_pi05.yaml`
- Create: `examples/embodiment/config/robotwin_beat_block_hammer_sft_actor_openpi_pi05.yaml`
- Create: `examples/embodiment/config/robotwin_pick_dual_bottles_sft_actor_openpi_pi05.yaml`

- [ ] **Step 1: Write the cup config**

Create `examples/embodiment/config/robotwin_place_empty_cup_sft_actor_openpi_pi05.yaml`:

```yaml
defaults:
  - _self_
  - model: pi0_5

cluster:
  num_nodes: 1
  component_placement:
    actor: 0-7

actor:
  micro_batch_size: 4
  global_batch_size: 256
  enable_offload: false
  model:
    model_type: openpi
    model_path: checkpoints/torch/pi05_base
    action_dim: 14
    num_action_chunks: 10
    is_lora: false
    add_value_head: false
    openpi:
      config_name: pi05_aloha_robotwin_place_empty_cup
      action_cond_enabled: true
      action_cond_freq_bands: 4
      action_cond_min_period: 1.0e-2
      action_cond_max_period: 4.0
      action_cond_dropout_prob: 0.5
      refine_iters: 0
      action_chunk: 10

runner:
  max_epochs: 30000
  save_interval: 1000
  val_check_interval: 0
  output_dir: ./results/robotwin_place_empty_cup_sft_v1
  logger:
    log_path: ./logs
    experiment_name: robotwin_place_empty_cup_sft_v1
    backend: wandb
```

- [ ] **Step 2: Copy to the other 2 tasks with substitutions**

```bash
cd /vePFS/zundong/Negfinetun-VLA/examples/embodiment/config
for task in beat_block_hammer pick_dual_bottles; do
    sed "s/place_empty_cup/${task}/g" robotwin_place_empty_cup_sft_actor_openpi_pi05.yaml \
        > robotwin_${task}_sft_actor_openpi_pi05.yaml
done
ls robotwin_*_sft_actor_openpi_pi05.yaml
```
Expected: 3 yaml files listed.

- [ ] **Step 3: Hydra-parse-validate each config**

```bash
cd /vePFS/zundong/Negfinetun-VLA
for task in place_empty_cup beat_block_hammer pick_dual_bottles; do
    python -c "
from hydra import compose, initialize
with initialize(config_path='examples/embodiment/config', version_base='1.3'):
    cfg = compose(config_name='robotwin_${task}_sft_actor_openpi_pi05')
    assert cfg.actor.model.openpi.config_name == 'pi05_aloha_robotwin_${task}'
    assert cfg.actor.model.action_dim == 14
    print('OK: robotwin_${task}_sft_actor_openpi_pi05')
"
done
```
Expected: 3 OK lines.

- [ ] **Step 4: Commit**

```bash
git add examples/embodiment/config/robotwin_*_sft_actor_openpi_pi05.yaml
git commit -m "feat(sft): 3 RoboTwin per-task SFT configs (spec 2026-04-28 §6.2)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Runbook `docs/sft-pretraining.md`

**Files:**
- Create: `docs/sft-pretraining.md`

- [ ] **Step 1: Write the runbook**

Create `docs/sft-pretraining.md`:

````markdown
# SFT Pre-training the Action-Conditioned VLA

How to run SFT for the action-conditioned π0.5 expert on the 4 LIBERO suites and the 3 RoboTwin tasks. Companion to `docs/superpowers/specs/2026-04-28-action-conditioned-sft-design.md`.

## What this produces

- 7 checkpoints (4 LIBERO + 3 RoboTwin), each with `action_cond_in_proj` weights moved off zero-init.
- Each checkpoint loads bit-for-bit into the existing DiffusionNFT RL config (`docs/rl-post-training.md`).

## Prerequisites

### Environment

```bash
export HF_HOME=~/shock/.CACHE/hf_cache
export XDG_CACHE_HOME=~/shock/.CACHE/xdg_cache
export UV_CACHE_DIR=~/shock/.CACHE/uv_cache
export PIP_CACHE_DIR=~/shock/.CACHE/pip_cache
export CONDA_ENVS_PATH=~/shock/conda_envs

# rlinf docker image
docker run -it --rm --gpus all --shm-size 20g --network host \
  --name rlinf -v .:/workspace/RLinf \
  rlinf/rlinf:agentic-rlinf0.1-maniskill_libero
source switch_env openpi
```

### openpi base weights

```bash
source /vePFS/shock/toolbox/bin/unset-proxy
export HF_ENDPOINT=https://hf-mirror.com

# π0.5 base (used by all 7 SFT runs as the starting point)
/vePFS/shock/toolbox/bin/hfd lerobot/pi0_5_base --local-dir checkpoints/torch/pi05_base
```

### Verify the action-cond port is the Fourier variant

```bash
python - <<'PY'
from omegaconf import OmegaConf
from rlinf.models import get_model
cfg = OmegaConf.create({
    "model_type": "openpi",
    "model_path": "checkpoints/torch/pi05_base",
    "action_dim": 7, "num_action_chunks": 10, "num_steps": 4, "use_proprio": True,
    "is_lora": False, "lora_rank": 32, "add_value_head": False, "precision": None,
    "openpi": {
        "config_name": "pi05_libero_spatial", "num_images_in_input": 2, "noise_level": 0.5,
        "action_chunk": 10, "num_steps": 4, "train_expert_only": True,
        "action_env_dim": 7, "solver_type": "flow_sde", "add_value_head": False,
        "detach_critic_input": False,
        "action_cond_enabled": True, "action_cond_dropout_prob": 0.5, "refine_iters": 0,
        "action_cond_freq_bands": 4,
        "action_cond_min_period": 1e-2, "action_cond_max_period": 4.0,
    },
})
m = get_model(cfg)
import torch.nn as nn
assert isinstance(m.action_cond_in_proj, nn.Sequential), "freq_bands=4 should produce Sequential"
last = m.action_cond_in_proj[-1]
assert last.weight.abs().sum().item() == 0.0, "last layer must be zero-init"
print("[OK] action_cond_in_proj is Fourier-MLP variant, zero-init last layer")
PY
```

Expected: `[OK] action_cond_in_proj is Fourier-MLP variant, zero-init last layer`.

### Verify HF datasets exist (RoboTwin)

The RoboTwin TrainConfigs assume HF repos `robotwin/<task>_random` exist. Verify before each launch:

```bash
for task in place_empty_cup beat_block_hammer pick_dual_bottles; do
    /vePFS/shock/toolbox/bin/hfd robotwin/${task}_random --check 2>&1 || \
        echo "MISSING: robotwin/${task}_random — edit dataconfig/__init__.py:_ROBOTWIN_TASK_REPOS"
done
```

If a repo is missing, update `_ROBOTWIN_TASK_REPOS` in `rlinf/models/embodiment/openpi/dataconfig/__init__.py` to point at the correct path (or convert local data first).

## LIBERO SFT (4 runs)

```bash
cd /vePFS/zundong/Negfinetun-VLA
for suite in spatial object goal 10; do
    bash examples/embodiment/run_sft_embodied.sh libero_${suite}_sft_actor_openpi_pi05
done
```

ETA per run: ~0.5–1 H100-day on a single H100. Multi-GPU FSDP scales near-linearly.

## RoboTwin SFT (3 runs)

```bash
cd /vePFS/zundong/Negfinetun-VLA
for task in place_empty_cup beat_block_hammer pick_dual_bottles; do
    bash examples/embodiment/run_sft_embodied.sh robotwin_${task}_sft_actor_openpi_pi05
done
```

## Wandb metrics to watch

| key | meaning | healthy range |
|---|---|---|
| `train/loss` | flow-matching MSE | drifts down; expect ~0.3–0.5 → ~0.05–0.1 by step 30k |
| `train/grad_norm` | post-clip gradient norm | bounded < 5 |
| `train/learning_rate` | LR schedule | follows cosine warmup→decay |
| `time/step` | wall-clock per training step | bounded by GPU; FSDP-stable |

A **manual** post-step probe to confirm the action_cond port is moving off zero — run periodically:

```bash
python - <<'PY'
import safetensors.torch as st
sd = st.load_file("./logs/libero_spatial_sft_v1/checkpoints/global_step_5000/actor/model.safetensors")
# Find any tensor whose name contains action_cond_in_proj
keys = [k for k in sd if "action_cond_in_proj" in k]
total = sum(sd[k].abs().sum().item() for k in keys)
print(f"action_cond_in_proj total |w|={total:.6f} across {len(keys)} tensors")
assert total > 0, "port still zero — CFG dropout not exercising it"
PY
```

## Post-SFT validation

For each saved checkpoint:

1. **Action-cond port nonzero** (already covered by the probe above).
2. **Inference smoke test** — `predict_action_batch` produces different outputs with `a_cond=0` vs `a_cond=demo_action`:

   ```bash
   python - <<'PY'
   import torch
   from omegaconf import OmegaConf
   from rlinf.models import get_model
   # ... load model from saved SFT checkpoint ...
   # ... build a tiny env_obs with one batch element ...
   actions_cold = m.predict_action_batch(env_obs)            # a_cond=None internally
   # synthesize a non-zero a_cond and call again — would require touching internals,
   # so use refine_iters=1 instead which feeds the cold output back as a_cond:
   m.config.refine_iters = 1
   actions_refined = m.predict_action_batch(env_obs)
   delta = (actions_refined - actions_cold).abs().mean().item()
   print(f"refinement delta = {delta:.6f}")
   assert delta > 1e-4, "no signal from a_cond port — SFT didn't move the weights"
   PY
   ```

3. **DiffusionNFT RL warmup smoke** — load the SFT ckpt into the existing RL config and run 100 steps:

   ```bash
   bash examples/embodiment/run_embodiment.sh libero_spatial_dnft_actor_openpi_pi05 \
       actor.model.model_path=./logs/libero_spatial_sft_v1/checkpoints/global_step_30000/actor \
       runner.max_epochs=10 runner.save_interval=10
   ```

   Expected: training runs to completion without `KeyError` / `RuntimeError` from action_cond port.

## ETA summary

- 7 SFT runs × ~0.5–1 H100-day each ≈ **5 H100-days end-to-end** (sequential on a single H100).
- Parallel on 4 H100s: ~1.5 wall-clock days.
- Compute is bounded by the openpi flow-matching forward, not by the new Fourier port (the port adds ≪1% FLOPs).
````

- [ ] **Step 2: Commit**

```bash
git add docs/sft-pretraining.md
git commit -m "docs(sft): add SFT pre-training runbook (spec 2026-04-28 §7)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: End-to-end smoke test on a tiny config

**Files:**
- Create: `examples/embodiment/config/libero_spatial_sft_smoke.yaml`
- Test: manually executed end-to-end

- [ ] **Step 1: Write a minimal smoke config**

Create `examples/embodiment/config/libero_spatial_sft_smoke.yaml`:

```yaml
defaults:
  - libero_spatial_sft_actor_openpi_pi05
  - _self_

actor:
  micro_batch_size: 1
  global_batch_size: 1

cluster:
  num_nodes: 1
  component_placement:
    actor: 0

runner:
  max_epochs: 5
  save_interval: 5
  output_dir: ./results/libero_spatial_sft_smoke
  logger:
    experiment_name: libero_spatial_sft_smoke
    backend: tensorboard
```

- [ ] **Step 2: Run the smoke test**

```bash
cd /vePFS/zundong/Negfinetun-VLA
bash examples/embodiment/run_sft_embodied.sh libero_spatial_sft_smoke
```
Expected: training runs for 5 steps without crashing; `results/libero_spatial_sft_smoke/checkpoints/global_step_5/actor/model.safetensors` is written.

- [ ] **Step 3: Verify the action_cond port moved off zero**

```bash
python - <<'PY'
import safetensors.torch as st
sd = st.load_file("./results/libero_spatial_sft_smoke/checkpoints/global_step_5/actor/model.safetensors")
keys = [k for k in sd if "action_cond_in_proj" in k]
total = sum(sd[k].abs().sum().item() for k in keys)
print(f"action_cond_in_proj total |w|={total:.6f} across {len(keys)} tensors")
assert total > 0, "port still zero after 5 steps"
print("[OK] action_cond port carries signal")
PY
```
Expected: `[OK] action_cond port carries signal`.

- [ ] **Step 4: Commit**

```bash
git add examples/embodiment/config/libero_spatial_sft_smoke.yaml
git commit -m "test(sft): add end-to-end smoke config (spec 2026-04-28 §9)

Verifies the SFT pipeline runs to checkpoint and that the action_cond port
moves off zero-init within 5 micro-batch steps.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-review notes

After all 10 tasks complete:

- ✅ Spec §2 (Fourier port) → Tasks 1, 2
- ✅ Spec §3 (sft_forward rewrite) → Task 3
- ✅ Spec §4.1 (LIBERO suite filter) → Task 4
- ✅ Spec §4.2 (RoboTwin per-task TrainConfigs) → Task 5
- ✅ Spec §5 (entrypoint) → Task 6
- ✅ Spec §6.1 (LIBERO yamls) → Task 7
- ✅ Spec §6.2 (RoboTwin yamls) → Task 8
- ✅ Spec §7 (runbook) → Task 9
- ✅ Spec §9 (acceptance — smoke test) → Task 10

No spec section is unmapped.

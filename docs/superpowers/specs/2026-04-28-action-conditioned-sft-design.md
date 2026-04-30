# SFT Pipeline for the Action-Conditioned VLA

**Status:** design (2026-04-28). Companion to `2026-04-24-action-conditioned-expert-and-oar-rollout-design.md` and `2026-04-25-diffusion-nft-loss-design.md`. Targets the policy class `OpenPi0ForRLActionPrediction` already landed on master.

## 0. Scope and goal

Build a working SFT pre-training path for the action-conditioned π0.5 expert in this repo, on **LIBERO (4 per-suite runs: `spatial`, `object`, `goal`, `10`)** and **RoboTwin (3 task runs: `place_empty_cup`, `beat_block_hammer`, `pick_dual_bottles`)** = **7 SFT runs total**. The output of each run is a checkpoint where:

1. The PaliGemma + Gemma-300M action expert match (or improve over) the upstream RLinf SFT checkpoint on a held-out chunk loss.
2. `action_cond_in_proj` weights have moved off zero-init and carry signal (verified by an inference smoke test where `a_cond` actually changes the output).
3. The checkpoint loads bit-for-bit into both the existing DiffusionNFT RL loop and the model-based-rollout pipeline being designed in parallel (Track A).

Out of scope:
- π0 (non-π0.5) SFT variants — every SFT run targets π0.5.
- Bundled-multi-task LIBERO SFT (we run one per-suite SFT instead — see §4 LIBERO suite filter knob).
- Co-training the world model.
- ManiSkill / MetaWorld / CALVIN / RoboCasa / Behavior SFT.

## 0.5. Why we need this at all

The RLinf-hosted SFT checkpoints (`RLinf/RLinf-Pi05-LIBERO-SFT`, `RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle`) load bit-for-bit because `action_cond_in_proj` is zero-init — but the user has flagged that those checkpoints "cannot be used" for our purpose. So we re-run SFT from scratch on each benchmark, starting from the upstream openpi base weights (`pi0_base` / `pi05_base`) and using the openpi flow-matching loss, but routed through our action-cond-aware velocity head so `action_cond_in_proj` learns non-trivial weights during this phase.

**Key invariant:** the SFT path must exercise `action_cond_in_proj` at every micro-batch, otherwise the port stays at zero and the downstream DiffusionNFT RL / WM rollout consumers carry no information through `a_cond`. Today's `sft_forward` does *not* exercise the port — fixing that is §2 of this spec.

---

## 1. Sub-deliverables

The spec covers five code changes plus seven SFT configs and a runbook:

| § | Deliverable | Type | LOC est. |
|---|---|---|---|
| 2 | `OpenPi0Config` Fourier-feature port revision | code | ~80 |
| 3 | `OpenPi0ForRLActionPrediction.sft_forward` rewrite | code | ~60 |
| 4a | `LeRobotLiberoDataConfig` extended with `suite: str` filter + 4 new per-suite TrainConfigs | code | ~120 |
| 4b | 3 new `pi05_aloha_robotwin_<task>` TrainConfigs (HF dataset paths assumed; verified at runbook §1) | code | ~90 |
| 5 | `train_sft_embodied.py` entrypoint | code | ~80 |
| 6 | 7 SFT configs: 4 × `libero_<suite>_sft_actor_openpi_pi05.yaml` + 3 × `robotwin_<task>_sft_actor_openpi_pi05.yaml` | config | ~60 each ≈ 420 |
| 7 | Runbook in `docs/sft-pretraining.md` | docs | ~200 |

Total: ~850 LOC + ~200 lines docs. Engineering ETA: ~7 days. Compute: ~0.5–1 H100-day × 7 runs ≈ **5 H100-days**.

---

## 2. §0 — Action-cond port revision (Fourier features)

**Why:** the current `action_cond_in_proj = nn.Linear(action_dim, expert_width)` is bandwidth-limited for low-dim, high-frequency action signals. Mirroring upstream's `action_time_mlp` recipe (sinusoidal expansion → 2-layer MLP) gives the projection direct access to multiple frequency bands and aligns our port with how upstream conditions on the scalar `t`.

### 2.1 New config knobs (in `OpenPi0Config`)

```python
action_cond_freq_bands: int = 0           # 0 = legacy bare-Linear (bit-for-bit fallback)
action_cond_min_period: float = 1e-2
action_cond_max_period: float = 4.0
```

Default `freq_bands=0` keeps the current Linear-only behavior, so every existing SFT checkpoint and zero-init test still passes. New SFT runs in this spec set `freq_bands=4`.

### 2.2 `__init__` change (in `OpenPi0ForRLActionPrediction.__init__`)

Replace the bare `nn.Linear` registration with a Fourier-aware module:

```python
if getattr(self.config, "action_cond_enabled", True):
    import torch.nn as _nn
    expert_w = self.action_in_proj.weight.shape[0]
    K = int(getattr(self.config, "action_cond_freq_bands", 0))
    if K == 0:
        # legacy: bare Linear, zero-init
        self.action_cond_in_proj = _nn.Linear(self.config.action_dim, expert_w)
        _nn.init.zeros_(self.action_cond_in_proj.weight)
        _nn.init.zeros_(self.action_cond_in_proj.bias)
    else:
        in_dim = self.config.action_dim * (1 + 2 * K)
        l1 = _nn.Linear(in_dim, expert_w)
        l2 = _nn.Linear(expert_w, expert_w)
        # zero-init the LAST layer so cold start emits zeros (bit-for-bit fallback property preserved)
        _nn.init.zeros_(l2.weight)
        _nn.init.zeros_(l2.bias)
        self.action_cond_in_proj = _nn.Sequential(l1, _nn.SiLU(), l2)
    # device/dtype align with the rest of the action expert
    self.action_cond_in_proj = self.action_cond_in_proj.to(
        dtype=self.action_in_proj.weight.dtype,
        device=self.action_in_proj.weight.device,
    )
```

### 2.3 New helper `_encode_action_cond`

```python
def _encode_action_cond(self, a: torch.Tensor) -> torch.Tensor:
    """Encode a clean action chunk into expert-width tokens.
    Adds sinusoidal Fourier features per dim when freq_bands > 0;
    falls back to bare projection when freq_bands == 0.
    """
    K = int(getattr(self.config, "action_cond_freq_bands", 0))
    if K == 0:
        return self.action_cond_in_proj(a)
    fraction = torch.linspace(0.0, 1.0, K, device=a.device)
    period = self.config.action_cond_min_period * (
        self.config.action_cond_max_period / self.config.action_cond_min_period
    ) ** fraction
    scale = (1.0 / period) * 2 * math.pi          # [K]
    proj = a.unsqueeze(-1) * scale                # [B,H,D,K]
    enc = torch.cat([a.unsqueeze(-1), proj.sin(), proj.cos()], dim=-1)  # [B,H,D,1+2K]
    enc = enc.flatten(-2, -1)                     # [B,H,D*(1+2K)]
    return self.action_cond_in_proj(enc)
```

### 2.4 `embed_suffix` change

Replace the call site at `openpi_action_model.py:1303`:

```python
# was: return self.action_cond_in_proj(a)
return self._encode_action_cond(a)
```

### 2.5 Bit-for-bit fallback property

`action_cond_freq_bands == 0` ⇒ `_encode_action_cond` is exactly `self.action_cond_in_proj(a)` ⇒ identical semantics to today. Existing checkpoints (zero-init) still load and produce zero `a_cond` embedding. The DiffusionNFT RL loop passes its tests unchanged.

### 2.6 Test

Add `tests/test_action_cond_fourier.py`:

```python
def test_freq_bands_zero_is_bit_for_bit():
    # build two models, freq_bands=0 vs old bare-Linear, identical seed → identical outputs
    ...

def test_freq_bands_four_zero_init_emits_zeros():
    # build with freq_bands=4, last-layer zero-init → output is exactly zeros
    ...

def test_freq_bands_four_post_step_nonzero():
    # one optimizer step on dummy SFT loss → action_cond_in_proj weights non-zero
    ...
```

---

## 3. §0.5 — `sft_forward` rewrite

**Why:** today `sft_forward` is `return super().forward(observation, actions)`. Upstream's `PI0Pytorch.forward` does *not* pass `action_cond` through `embed_suffix`. So today's SFT path leaves `action_cond_in_proj` untouched. We rewrite it to run the flow-matching loss directly through our `get_velocity` (which already accepts `action_cond`).

### 3.1 New `sft_forward` (in `OpenPi0ForRLActionPrediction`)

```python
def sft_forward(self, data, **kwargs):
    """Flow-matching SFT loss, action-cond-aware.

    Computes v_target = noise - x_0, v_pred = get_velocity(obs, x_t, t, action_cond),
    where action_cond is sampled via CFG dropout (see _sample_action_cond).
    Returns the per-batch MSE loss (mean over B,H,D).
    """
    observation = data["observation"]
    actions = data["actions"]                              # [B, H, action_dim], clean x_0

    # CFG-dropped action_cond — exercises action_cond_in_proj
    action_cond = self._sample_action_cond(actions)        # see line 231

    # input transform → openpi Observation object
    obs_dict = self.input_transform(data, transpose=False)
    obs_obj  = _model.Observation.from_dict(obs_dict)
    images, img_masks, lang_tokens, lang_masks, state = (
        self._preprocess_observation(obs_obj, train=True)
    )
    device = actions.device
    images = [img.to(device) for img in images]
    img_masks = [m.to(device) for m in img_masks]
    state = state.to(device)

    # sample t ∼ Beta or U[ε, 1−ε] (match upstream openpi for fairness)
    B = actions.shape[0]
    t = sample_beta(alpha=1.5, beta=1.0, bsize=B, device=device).clamp(1e-3, 1 - 1e-3)
    noise = torch.randn_like(actions)
    t_b = t[:, None, None]
    x_t = (1.0 - t_b) * actions + t_b * noise              # standard flow-matching path
    v_target = noise - actions                             # standard flow-matching velocity

    # action-cond-aware velocity prediction (action_cond plumbed through embed_suffix)
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

### 3.2 Reuses existing infra

- `_sample_action_cond` already on master (line 231).
- `get_velocity_for_oar(observation, x_t, t, *, action_cond, compute_grad)` already on master (added in commit `e25b8d7`).
- `_preprocess_observation`, `input_transform` already on master.
- `sample_beta` exists in upstream openpi (`pi0_pytorch.py:45`).

So the rewrite is an interception, not new infrastructure.

### 3.3 Equivalence to upstream openpi when `action_cond_freq_bands == 0` AND `action_cond_dropout_prob == 1.0`

When freq_bands=0 (Linear-only port, zero-init) AND dropout_prob=1.0 (always-zero a_cond), `_encode_action_cond` produces zero embeddings ⇒ `embed_suffix` emits the same H zero tokens as upstream's "no a_cond" baseline ⇒ the loss is mathematically equivalent to upstream's flow-matching loss. **This is the "ablation control" config** for sanity-checking the SFT pipeline against upstream numbers.

---

## 4. §1 — Data loaders

`FSDPSftWorker.build_dataloader` at `fsdp_sft_worker.py:60` already routes through `openpi_data_loader.create_data_loader` driven by `self.cfg.actor.model.openpi.config_name`. The base `pi05_libero` and `pi0_aloha_robotwin` configs exist. Two changes are required.

### 4.1 LIBERO suite filter knob (§4a)

The HF dataset `physical-intelligence/libero` bundles all 4 suites in one repo. To run per-suite SFT we extend `LeRobotLiberoDataConfig` with a `suite: str | None = None` field that, when set, filters episodes whose `task` field matches the suite name (LIBERO task strings are prefixed with the suite name, e.g. `libero_spatial:pick_up_the_black_bowl_...`).

```python
# dataconfig/libero_dataconfig.py
@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    extra_delta_transform: bool = False
    suite: str | None = None              # NEW: "spatial" | "object" | "goal" | "10" | None (=all)

    @override
    def create(self, assets_dirs, model_config) -> DataConfig:
        cfg = super().create(...)         # existing logic
        if self.suite is not None:
            # apply a one-shot HF dataset filter on `task` prefix
            cfg = dataclasses.replace(cfg, episode_filter=lambda ep: ep["task"].startswith(f"libero_{self.suite}"))
        return cfg
```

If openpi's `DataConfig` does not natively support `episode_filter`, we add a small wrapper transform that drops out-of-suite samples after batching (less efficient but correct). Implementation choice deferred to the PR — both are bounded.

Then add 4 new TrainConfigs to `dataconfig/__init__.py`, mirroring the existing `pi05_libero` block:

```python
for suite in ["spatial", "object", "goal", "10"]:
    _CONFIGS.append(TrainConfig(
        name=f"pi05_libero_{suite}",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_libero/assets"),
            extra_delta_transform=False,
            suite=suite,
        ),
        batch_size=256, ...   # mirror pi05_libero settings
    ))
```

### 4.2 RoboTwin per-task TrainConfigs (§4b)

The repo has `pi0_aloha_robotwin` for `place_empty_cup_random` only. Add 3 pi05 TrainConfigs for the 3 in-scope tasks. Dataset names are **assumed** to follow `robotwin/<task_name>_random` based on the one existing config; the runbook §1 verifies on HF before SFT launch.

```python
ROBOTWIN_TASK_REPOS = {
    "place_empty_cup":   "robotwin/place_empty_cup_random",      # verified
    "beat_block_hammer": "robotwin/beat_block_hammer_random",    # ASSUMED
    "pick_dual_bottles": "robotwin/pick_dual_bottles_random",    # ASSUMED
}
for task, repo_id in ROBOTWIN_TASK_REPOS.items():
    _CONFIGS.append(TrainConfig(
        name=f"pi05_aloha_robotwin_{task}",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotAlohaDataConfig(
            repo_id=repo_id,
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir=f"checkpoints/torch/pi05_robotwin_{task}/assets"),
            extra_delta_transform=True,
        ),
        pytorch_weight_path="checkpoints/torch/pi05_base",
        batch_size=256, optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999, num_workers=8, num_train_steps=30_000,
    ))
```

If the assumed repos do not exist, runbook §1 instructs the user to `hf datasets list robotwin/` and adjust `ROBOTWIN_TASK_REPOS` before running SFT for that task.

---

## 5. §2 — Entrypoint `train_sft_embodied.py`

Mirror `examples/embodiment/train_embodied_agent.py` but dispatch to `SFTRunner` instead of the embodied RL runner. Single hydra entrypoint. Pseudocode:

```python
@hydra.main(config_path="config", config_name=None, version_base="1.3")
def main(cfg: DictConfig):
    cluster = build_cluster(cfg)
    placement = HybridComponentPlacement(cfg, cluster)

    actor = FSDPSftWorker.options(
        placement=placement.get_placement("actor"),
    ).remote(cfg)

    runner = SFTRunner(cfg=cfg, actor=actor, run_timer=ScopedTimer())
    runner.init_workers()
    runner.run()
```

Run script `examples/embodiment/run_sft_embodied.sh` parallels `run_embodiment.sh` but invokes the new entrypoint.

---

## 6. §3 — SFT configs (7 total)

### 6.1 LIBERO per-suite — template `libero_<suite>_sft_actor_openpi_pi05.yaml`

```yaml
# libero_spatial_sft_actor_openpi_pi05.yaml
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
      config_name: pi05_libero_spatial      # ← varies per suite
      action_cond_enabled: true
      action_cond_freq_bands: 4              # NEW
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

The other 3 LIBERO configs (`object`, `goal`, `10`) are identical save for `openpi.config_name` (`pi05_libero_object`, …), `output_dir`, and `experiment_name`.

### 6.2 RoboTwin per-task — template `robotwin_<task>_sft_actor_openpi_pi05.yaml`

Identical to 6.1 except: `actor.model.action_dim: 14`, `openpi.config_name: pi05_aloha_robotwin_<task>`, `output_dir: ./results/robotwin_<task>_sft_v1`, `experiment_name: robotwin_<task>_sft_v1`.

The 3 task names: `place_empty_cup`, `beat_block_hammer`, `pick_dual_bottles`.

---

## 7. §4 — Runbook (`docs/sft-pretraining.md`)

Required sections (full content drafted in the implementation PR, not in this spec):

1. **Prerequisites** — openpi base weights download, HF dataset download, env vars (`HF_HOME`, etc.).
2. **Action-cond port verification** — `python -c "..."` snippet to confirm `action_cond_freq_bands=4` and `action_cond_in_proj` is the Sequential variant.
3. **HF dataset verify (RoboTwin)** — `hfd robotwin/beat_block_hammer_random --check` etc.; if a repo doesn't exist, edit `ROBOTWIN_TASK_REPOS` to point at the correct HF path (or convert local data first).
4. **LIBERO SFT commands (4 runs)** — one per suite:
   ```bash
   for suite in spatial object goal 10; do
     bash examples/embodiment/run_sft_embodied.sh libero_${suite}_sft_actor_openpi_pi05
   done
   ```
5. **RoboTwin SFT commands (3 runs)** — one per task:
   ```bash
   for task in place_empty_cup beat_block_hammer pick_dual_bottles; do
     bash examples/embodiment/run_sft_embodied.sh robotwin_${task}_sft_actor_openpi_pi05
   done
   ```
6. **Wandb metrics to watch** — `train/loss` (decreasing), `train/grad_norm` (bounded), and a periodic action_cond-port-weight-norm metric to confirm the port is moving off zero.
7. **Validation** — post-SFT, run the existing `test_action_cond_fourier.py::test_freq_bands_four_post_step_nonzero` against the saved checkpoint; run a one-step inference smoke test (compute `predict_action_batch` with `a_cond=0` then with `a_cond=demo_action` and verify outputs differ).
8. **ETA** — ~0.5–1 H100-day per run × 7 runs ≈ **5 H100-days end-to-end**.

---

## 8. Risks and mitigations

| risk | likelihood | mitigation |
|---|---|---|
| Assumed RoboTwin HF repos (`beat_block_hammer_random`, `pick_dual_bottles_random`) don't exist | **high** | runbook §1 has a verify step; if missing, point at user-converted local LeRobot dataset, OR drop to 1 RoboTwin run (cup only) and re-open scope |
| LIBERO suite filter knob mis-applied if openpi `DataConfig` lacks `episode_filter` | medium | spec lists two implementation paths (native filter vs post-batch wrapper); PR picks whichever matches the openpi version pinned in this repo |
| Per-suite LIBERO undertrains because each suite has fewer episodes | medium | bump `num_train_steps` to 60_000 per suite if held-out loss plateaus high; or fall back to bundled SFT then suite-specific finetune |
| `_encode_action_cond` introduces FSDP wrap-name collision | low | the helper is a method, not a module; FSDP only wraps `nn.Module` named children; the new `Sequential` inherits `_fsdp_wrap_name` from the existing rename loop at `openpi_action_model.py:223` |
| `action_cond_dropout_prob=0.5` too aggressive for SFT (model never sees full a_cond) | medium-high | runbook §6 watches the action_cond-port-weight-norm; if growth is sluggish, drop to 0.3 |
| Upstream `sample_beta` import path drifts | low | re-implement inline if needed (3-line function) |
| `action_cond_freq_bands=4` breaks gradient checkpointing | low | the helper is pure tensor math, no module state; no checkpoint hook needed |
| 7-run sequential schedule on a single H100 takes ~5 days wall-clock | medium | parallelize on N GPUs/nodes via FSDP; document multi-node command in runbook |

---

## 9. Acceptance criteria

A run is "done" when all of the following hold:

1. `train/loss` plateaus on LIBERO (resp. RoboTwin); ETA-checked at ~30k steps.
2. `tests/test_action_cond_fourier.py` passes against the saved checkpoint.
3. `predict_action_batch` smoke test shows non-zero output difference between `a_cond=0` and `a_cond=demo_action`.
4. The saved checkpoint loads cleanly (no missing/unexpected keys) into an existing DiffusionNFT RL config (`libero_spatial_dnft_actor_openpi_pi05.yaml`).
5. A 100-step DiffusionNFT RL warmup using the new SFT checkpoint runs to completion (no wire breakage).

## 10. Out-of-scope follow-ups

- π0 (non-π0.5) variants of all 7 SFT runs.
- Bundled multi-task LIBERO SFT (only re-open if per-suite shows degeneracy).
- RoboTwin tasks beyond the in-scope 3 (need new HF datasets / TrainConfigs).
- ManiSkill / MetaWorld / CALVIN / RoboCasa SFT.
- Co-training the WM (Track A) on top of the SFT data loader.
- Eval-during-SFT against the LIBERO/RoboTwin sim (currently `val_check_interval=0`).

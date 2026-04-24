# Design: Action-Conditioned Expert + Offline (o, a, r) Rollout Collection

- **Date**: 2026-04-24
- **Author**: iterated with user (xiaokejiang2026@gmail.com)
- **Scope**: Two coupled changes in `Negfinetun-VLA` (RLinf / π-StepNFT fork)
  1. Change the π0 / π0.5 action-expert from `a = f(o)` to `a = f(o, a_in)` via new conditioning-action tokens in the suffix, with CFG-style dropout at training and a two-pass refinement option at inference.
  2. Add an offline rollout-collection path that writes `(observation, action, reward)` transitions to disk per rollout epoch, alongside the existing in-memory PPO/NFT buffer.
- **Status**: awaiting user review (no code changes yet)

---

## 1. Motivation

The current action-expert (upstream `PI0Pytorch.embed_suffix`, used through `rlinf/models/embodiment/openpi/openpi_action_model.py`) takes only observation inputs (images, language, state) plus the flow-matching noisy action `x_t` and timestep. There is no channel for injecting an external *clean* action as conditioning.

The user wants:

- An explicit second input channel so the policy signature changes from `a = f(o)` to `a = f(o, a_in)`.
- At inference, the ability to run a **refinement pass**: cold pass with `a_in = 0` → produces `a_0`; refinement pass with `a_in = a_0` → produces `a_1`; optionally iterate.
- Training data that pairs observations with the actually-taken policy actions and their rewards, saved to disk, for later offline training, replay, or analysis.

Both changes share the same conceptual axis (the policy gains explicit coupling between observations and actions) and are naturally delivered together.

---

## 2. Goals and non-goals

### Goals
- **G1**: New architectural port `a_cond` on the action-expert, integrated into the existing PaliGemma-with-expert attention layout for both π0 (concat-MLP timestep path) and π0.5 (adaRMS timestep path).
- **G2**: The modified expert is strictly backward compatible at init: loading an existing SFT checkpoint without the new weights produces numerically identical output to the unmodified model.
- **G3**: A training regime (CFG-style dropout on `a_cond`) that lets a single checkpoint serve both `a_in = 0` (legacy one-shot) and `a_in ≠ 0` (refinement) modes.
- **G4**: An inference mode that runs the refinement pass as a post-process around the existing flow-matching denoising loop, without touching the solver code.
- **G5**: A parallel rollout collection path that writes atomic `(obs, executed_action, reward)` records per chunk step to disk, controlled by a single config flag, off by default.
- **G6**: Offline files readable by a plain PyTorch dataloader with no dependency on the running training process (i.e., tokenizer-agnostic, task-id-agnostic as much as possible).

### Non-goals
- No change to the flow-matching solvers (`euler`, `dpm`, `ddim`, `flow_sde`, `flow_noise`, `flow_grpo`, `flow_cps`, `dance`).
- No change to the value head, PPO/NFT loss, or advantage computation.
- No modification to the upstream `openpi` pip package (`/vePFS/shock/openpi/src/openpi/models_pytorch/pi0_pytorch.py`). The action-expert changes are implemented by subclassing inside this repo.
- No change to the rollout-to-actor wire format or the in-memory `EmbodiedRolloutResult` schema; the offline collector is strictly additive.
- No replay server, no streaming queue: this iteration writes plain `.pt` shards and stops there.

---

## 3. Action-expert architecture change

### 3.1 Current layout (baseline)

`PI0Pytorch.embed_suffix(state, noisy_actions, timestep)` in the upstream openpi library produces suffix tokens:

| Variant | Tokens | `att_masks` | Time conditioning |
|---|---|---|---|
| π0    | `[state(1) \| action_time(H)]` | `[1 \| 1, 0×(H−1)]` | concat `action_in_proj(x_t) ⊕ sinusoidal(t)` → MLP |
| π0.5  | `[action(H)]`                  | `[1, 0×(H−1)]`      | adaRMS cond from `time_mlp(sinusoidal(t))` |

`action_in_proj: Linear(action_dim, expert_width)` embeds the noisy action `x_t`.

### 3.2 Modified layout

Add one new linear projection and one new token block. Choice (a) = **A1** (confirmed): place `a_cond` between `state` and `action_time` for π0.

| Variant | Tokens | `att_masks` |
|---|---|---|
| π0    | `[state(1) \| a_cond(H) \| action_time(H)]` | `[1 \| 1, 0×(H−1) \| 1, 0×(H−1)]` |
| π0.5  | `[a_cond(H) \| action(H)]`                  | `[1, 0×(H−1) \| 1, 0×(H−1)]`      |

Semantics:

- `a_cond` tokens form their own attention block, placed after `state` (π0) or at the head of the suffix (π0.5).
- `action_time` attends to prefix + `state` + `a_cond` + itself.
- `a_cond` attends to prefix + (state, π0 only) + itself. **One-way conditioning**: `a_cond` does **not** attend to `action_time`. (Choice c = confirmed.) The refinement-loop semantics live in the *outer* inference loop, not inside a single forward pass.

### 3.3 New parameters

Added to the modified expert's `__init__`:

```python
self.action_cond_in_proj = nn.Linear(action_dim, action_expert_config.width)
nn.init.zeros_(self.action_cond_in_proj.weight)
nn.init.zeros_(self.action_cond_in_proj.bias)
```

**Zero-init** (choice b = **B1** confirmed). Consequence: the embedded `a_cond` tokens are zero for any input at initialization, so the modified model's output is numerically identical to the baseline when loaded from any unmodified SFT checkpoint. This satisfies goal G2 and removes the need for backward-compatibility shims.

### 3.4 Implementation location

Done via subclass inside this repo, so the pip-installed upstream openpi library is never patched.

- File to modify: `rlinf/models/embodiment/openpi/openpi_action_model.py`.
- Class `OpenPi0ForRLActionPrediction` already subclasses `PI0Pytorch`. Add:
  - `__init__`: call super, then register `self.action_cond_in_proj` with zero-init.
  - Override `embed_suffix(self, state, noisy_actions, timestep, action_cond=None)`:
    - When `action_cond is None`, default to `torch.zeros_like(noisy_actions)` (same shape).
    - Project through `action_cond_in_proj` to get `a_cond_emb` of shape `[B, H, expert_width]`.
    - Build the extended suffix and att_masks as in section 3.2.
- Update all internal call sites that pass through `embed_suffix` (there are four in `openpi_action_model.py`: around lines 649, 1047, 1071, 1116). Each call site either forwards an existing `action_cond` kwarg or passes `None`.

### 3.5 Training regime (CFG-style dropout)

In the existing training forward path (`OpenPi0ForRLActionPrediction.default_forward` at `openpi_action_model.py:366` and `sft_forward` at `:361` which delegates to the upstream `PI0Pytorch.forward`):

- `clean_action` here means the dataset / rollout action chunk that would otherwise be the flow-matching target — shape `[B, H, action_dim]`. It is already available in the forward's kwargs (the upstream signature is `PI0Pytorch.forward(observation, actions, noise=None, time=None)`).
- With probability `p_cond_dropout` (config, default `0.5`), set `action_cond = torch.zeros_like(clean_action)`.
- Otherwise, set `action_cond = clean_action.detach()` — the ground-truth action chunk detached from the graph, so gradients flow into `action_cond_in_proj` via the suffix tokens but not into the data.
- The flow-matching / NFT / PPO loss is unchanged. Only the forward signature is augmented.

Rationale for 50% dropout: a single checkpoint must serve both `a_in = 0` (cold pass at inference) and `a_in ≠ 0` (refinement pass). Standard CFG dropout rate is 50% and behaves well here.

### 3.6 Inference regime (refinement loop)

The existing `sample_actions` (`openpi_action_model.py:626`) runs K denoising steps producing `a_0`. To obtain the refinement `a_1`, we wrap this call rather than modifying the solver:

```python
# new method on OpenPi0ForRLActionPrediction
def sample_actions_refine(self, observation, num_refine_iters=1, **kwargs):
    a = self.sample_actions(observation, action_cond=None, **kwargs)["actions"]
    history = [a]
    for _ in range(num_refine_iters):
        outputs = self.sample_actions(observation, action_cond=a.detach(), **kwargs)
        a = outputs["actions"]
        history.append(a)
    return {"actions": a, "history": history}
```

- `sample_actions` gains an `action_cond: Optional[torch.Tensor]` kwarg (default `None`, preserves current behavior).
- At the outer caller (`predict_action_batch` at `:572`), a config flag `actor.model.openpi.refine_iters` (default `0`) switches between legacy one-shot and the refinement loop.
- Each refinement iteration is a *full* K-step denoising; cost is `(num_refine_iters + 1) × K` expert forwards instead of `K`.

### 3.7 Config additions

In `examples/embodiment/config/model/pi0.yaml` and `.../pi0_5.yaml`:

```yaml
openpi:
  # existing fields unchanged
  action_cond:
    enabled: true             # false => a_cond tokens never embedded (equivalent to baseline)
    dropout_prob: 0.5         # CFG-style dropout at training
    refine_iters: 0           # 0 => legacy single pass; 1 => one refinement pass; N => N refinements
```

`enabled: false` short-circuits the extra token block entirely (no `a_cond_in_proj` forward, no extra suffix tokens). This gives a clean kill-switch and makes ablations trivial.

---

## 4. Offline rollout collection — design

### 4.1 Goal

At every chunk step in the training rollout, simultaneously capture the triple `(observation, executed_action, reward)` and write the per-epoch aggregation to disk as a PyTorch shard, one per rollout-worker rank. Off by default; controlled by a config block.

### 4.2 Approach — β (parallel collector in rollout worker)

Add a second data struct `OARStepResult` populated inside `MultiStepRolloutWorker.generate()` (`rlinf/workers/rollout/hf/huggingface_worker.py:223`). The fields are aliases into the same tensors that are already produced for `ChunkStepResult` — no tensor copies, no extra GPU traffic. At the end of each rollout epoch, the rollout worker torch-saves its accumulated `OARRolloutBuffer` to a shard file.

### 4.3 Per-step schema

```python
@dataclass(kw_only=True)
class OARStepResult:
    # observation — raw env outputs, tokenizer-agnostic
    main_images: torch.Tensor            # [B, V, H_img, W_img, 3] uint8
    wrist_images: Optional[Tensor]       # [B, V, H_img, W_img, 3] uint8  (None if env has no wrist cam)
    state: torch.Tensor                  # [B, state_dim] fp32
    task_descriptions: list[str]         # length B; raw task strings, not tokens
    # action
    executed_action: torch.Tensor        # [B, H, action_dim] fp32  (post-intervene, matches forward_inputs["action"])
    # reward & termination
    reward: torch.Tensor                 # [B, H] fp32             (per env sub-step inside the chunk)
    done: torch.Tensor                   # [B, H] bool
    terminations: torch.Tensor           # [B, H] bool
    truncations: torch.Tensor            # [B, H] bool
    # auxiliary
    task_ids: Optional[Tensor]           # [B] int64
    success_once: Optional[Tensor]       # [B] bool
```

Choice E1 (confirmed): save the **executed** (post-intervene) action. If the intervene mechanism is inactive — the common case in pure on-policy RL — this is identical to the pure policy output.

Granularity (confirmed): per chunk-step (one record = 1 obs + H actions + H rewards). Flattening to H individual transitions would duplicate the obs and misrepresent the chunk policy.

### 4.4 Per-epoch buffer

```python
@dataclass(kw_only=True)
class OARRolloutBuffer:
    main_images: list[torch.Tensor]      = field(default_factory=list)
    wrist_images: list[Optional[Tensor]] = field(default_factory=list)
    state: list[torch.Tensor]            = field(default_factory=list)
    task_descriptions: list[list[str]]   = field(default_factory=list)
    executed_action: list[torch.Tensor]  = field(default_factory=list)
    reward: list[torch.Tensor]           = field(default_factory=list)
    done: list[torch.Tensor]             = field(default_factory=list)
    terminations: list[torch.Tensor]     = field(default_factory=list)
    truncations: list[torch.Tensor]      = field(default_factory=list)
    task_ids: list[Optional[Tensor]]     = field(default_factory=list)
    success_once: list[Optional[Tensor]] = field(default_factory=list)

    def append(self, step: OARStepResult) -> None: ...
    def to_dict(self) -> dict[str, torch.Tensor | list]: ...  # stacks along time axis T
```

Shapes after `to_dict()` (leading axis `T = rollout_epoch * n_chunk_steps`):

| Field | Shape |
|---|---|
| `main_images` | `[T, B, V, H_img, W_img, 3]` uint8 |
| `wrist_images` | same or `None` |
| `state` | `[T, B, state_dim]` fp32 |
| `task_descriptions` | nested list, shape `[T][B]` of str |
| `executed_action` | `[T, B, H, action_dim]` fp32 |
| `reward` | `[T, B, H]` fp32 |
| `done` / `terminations` / `truncations` | `[T, B, H]` bool |
| `task_ids` | `[T, B]` int64 (or absent) |
| `success_once` | `[T, B]` bool (or absent) |

No off-by-one: this buffer does **not** store the bootstrap tail observation that `EmbodiedRolloutResult` retains for GAE. Offline consumers don't need it.

### 4.5 Integration points

In `rlinf/workers/rollout/hf/huggingface_worker.py`:

- `__init__`: read `cfg.rollout.offline_save` block; set `self.offline_save_enabled`, `self.offline_save_dir`, `self.offline_save_start_step`.
- `generate()`: before the rollout-epoch loop, if enabled, instantiate `self.oar_buffer_list = [OARRolloutBuffer() for _ in range(num_pipeline_stages)]`.
- Immediately after each `predict(...)` inside the chunk loop, build an `OARStepResult` from:
  - `env_output["obs"]` → multimodal obs fields
  - `actions` (the tensor returned by `predict`) → `executed_action` (note: this is after `update_intervene_actions` merge because `forward_inputs["action"]` already carries the merged chunk)
  - `env_output["rewards"] / dones / terminations / truncations` (handled via `get_dones_and_rewards`)
  - `env_output.get("task_ids")`, `env_output.get("success_once")`
- Append to `self.oar_buffer_list[stage_id]`.
- After the outer rollout-epoch loop, if enabled, call `self._save_oar_shards(global_step)`:

```python
def _save_oar_shards(self, global_step: int) -> None:
    out_dir = (
        Path(self.offline_save_dir) / f"global_step_{global_step}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for stage_id, buf in enumerate(self.oar_buffer_list):
        shard_path = out_dir / f"rank_{self._rank}_stage_{stage_id}.pt"
        torch.save(buf.to_dict(), shard_path)
    # one metadata.json per step, written by rank 0 only, idempotent
    if self._rank == 0:
        self._write_metadata_json(out_dir)
```

No Ray coordinator is needed: each worker writes its own shards to a distinct path. `metadata.json` records shapes, dtypes, `T`, `B`, `H`, `action_dim`, and `action_cond` config, written exactly once per step by rank 0.

### 4.6 On-disk layout

```
<output_dir>/
  global_step_<N>/
    rank_0_stage_0.pt
    rank_0_stage_1.pt
    rank_1_stage_0.pt
    ...
    metadata.json
  global_step_<N+save_interval>/
    ...
```

`<output_dir>` defaults to `${runner.output_dir}/rollouts` when `runner.output_dir` exists, else `./data/rollouts/<timestamp>`.

Each `.pt` file is a dict produced by `OARRolloutBuffer.to_dict()`. Images saved as uint8 (no float widening). `torch.save` default (pickle, no compression) — fast enough and the bottleneck is disk, not CPU.

### 4.7 Reader

Not part of this spec, but for reference the intended consumer is a plain Dataset:

```python
# offline dataloader sketch (separate follow-up work)
class OARDataset(torch.utils.data.Dataset):
    def __init__(self, run_dir: str): ...
    def __len__(self): ...                # T * B across all shards
    def __getitem__(self, idx): ...       # one (obs, action[H, action_dim], reward[H], ...) record
```

### 4.8 Config additions

New block in every embodiment run config that uses the HF rollout worker:

```yaml
rollout:
  offline_save:
    enabled: false                                # default off
    output_dir: ${runner.output_dir}/rollouts     # interpolated
    start_step: 0                                 # skip first N global training steps
```

Default `enabled: false` preserves current behavior for all existing experiments.

---

## 5. End-to-end data flow after the change

```
 EnvWorker.interact                               MultiStepRolloutWorker.generate
 ────────────                                     ─────────────────────────────
 reset → obs  ──────── env_channel ──────────►   recv_env_output
                                                  │
                                                  │ preprocess_env_obs(obs)
                                                  │
                                                  ▼
                                           sample_actions(..., action_cond=None)  ◄─── flow matching, K steps
                                                  │     (optional: refine_iters loop wraps this)
                                                  ▼
                                           actions = x_0  [B, H, action_dim]
                                                  │
                                                  ├───► forward_inputs["action"]  ──► ChunkStepResult (in-memory, PPO buffer)
                                                  │
                                                  └───► OARStepResult.executed_action  ──► OARRolloutBuffer (offline)
                                                  │
 env.chunk_step  ◄──── rollout_channel ─────────  send_chunk_actions
  (H sub-steps)
  rewards, dones ──── env_channel ──────────►   get_dones_and_rewards → append to both buffers

 (end of rollout epoch)
                                                  send_rollout_batch  ──── actor_channel ──► Actor (PPO/NFT update)
                                                  _save_oar_shards(global_step)           ──► disk (new)
```

---

## 6. Testing plan

### 6.1 Architecture-change tests (unit)
- **T1**: After `OpenPi0ForRLActionPrediction.__init__`, check `action_cond_in_proj.weight.abs().sum() == 0` and bias is zero.
- **T2**: Load an existing SFT checkpoint (e.g. `RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT`) into the modified arch. Pick any observation batch; run `sample_actions(action_cond=None)` and `sample_actions(action_cond=zeros)` against a baseline model loaded from the upstream unmodified `PI0Pytorch`. Assert `max_abs_diff < 1e-5`.
- **T3**: With `action_cond_in_proj` weights set to a random nonzero value and a fixed seed, `sample_actions(action_cond=a)` and `sample_actions(action_cond=b)` with `a ≠ b` produce different outputs — sanity check that the new port actually influences the model.
- **T4**: Attention-mask shape test: the returned `att_masks` from the modified `embed_suffix` has length `1 + H + H` (π0) or `H + H` (π0.5), matching the new layout.

### 6.2 Training-path smoke tests
- **T5**: Run one training step on LIBERO-spatial (pi0 config) with `action_cond.enabled=true`, `dropout_prob=0.5`. Loss does not NaN. Gradients on `action_cond_in_proj.weight` are nonzero on the non-dropped-out minibatches.
- **T6**: Same with `action_cond.enabled=false` — check no extra parameters show up in the FSDP unit and the forward path is byte-for-byte the baseline (`assert equal` on model output).

### 6.3 Inference refinement tests
- **T7**: With `refine_iters=0`, `sample_actions_refine` returns the same action as the baseline `sample_actions` (modulo noise seed).
- **T8**: With `refine_iters=1`, check that `history` has length 2 and `history[1] ≠ history[0]` after a few training steps (before training, they will be equal due to zero-init).

### 6.4 Offline rollout collection tests
- **T9**: With `offline_save.enabled=false`, no directory is created; no shards on disk. The existing rollout-to-actor wire is byte-identical to baseline.
- **T10**: With `offline_save.enabled=true`, after one full rollout epoch:
  - Directory `global_step_<N>/` exists with one `.pt` per rank × stage.
  - `metadata.json` exists, written exactly once (not duplicated across ranks).
  - Loading a shard yields a dict with the expected keys and shapes; `executed_action.shape == [T, B, H, action_dim]`, dtypes match §4.4.
- **T11**: Assert `executed_action` in the shard equals `forward_inputs["action"]` in the in-memory PPO buffer for the same step (sanity: the two paths see the same tensor).
- **T12**: Run 2 consecutive training steps with `start_step=1`; verify that `global_step_0/` is absent and `global_step_1/` is present.

### 6.5 Multi-rank / multi-stage test (integration)
- **T13**: Run with `rollout.pipeline_stage_num=2` and 2 rollout workers. After one step, there are exactly `2 × 2 = 4` shard files plus one `metadata.json`; shapes are consistent.

---

## 7. Rollout / migration plan

1. Land the architecture change with defaults preserving baseline:
   - `action_cond.enabled: false` in all existing configs.
   - Zero-init guarantees loaded SFT checkpoints reproduce baseline bit-for-bit.
2. Run T1–T7 on LIBERO-spatial (fastest embodiment env).
3. Enable `action_cond.enabled: true` in one experimental config (`libero_spatial_ppo_openpi_pi05.yaml`) and train one short run to verify the dropout path learns without NaN.
4. Land the offline rollout collector with `offline_save.enabled: false`.
5. Run T9–T13 and one full LIBERO training with `offline_save.enabled: true` to validate on-disk artifacts.
6. Separately (follow-up, not this spec): write the `OARDataset` reader and a first pass that trains the new `action_cond_in_proj` on the collected offline data.

No existing training recipe is affected unless it explicitly opts in.

---

## 8. Open items / out of scope

- **Reader / offline dataloader** — deferred, referenced in §4.7 as a sketch.
- **Compression / sharding by size** — not in this iteration; simple per-epoch `.pt` shards are sufficient for early experiments. Swap to webdataset-tar or parquet when disk size becomes a constraint.
- **Per-env-step flattening** — deliberately not supported. If later needed for a non-chunk policy, add a post-processing utility that expands the chunk-level record into H sub-records.
- **Disk-persisted *policy* action vs executed action split (E2)** — deferred; add a `policy_action` field later if/when intervene is turned on in a study.
- **Refinement past one iteration** — `refine_iters > 1` is supported by the loop in §3.6 but has no training-time analog; CFG dropout trains only for the single-refinement regime. If iterated refinement becomes interesting, we will revisit the training schedule then.
- **Value head interaction with `a_cond`** — the value head is unchanged; the modified suffix tokens still feed through the same path, so `value_after_vlm` (π0.5) continues to work. If `a_cond` turns out to distort V(s), we can add a separate `value_action_cond` flag later.

---

## 9. File-by-file change list (preview for the implementation plan)

- `rlinf/models/embodiment/openpi/openpi_action_model.py`
  - `OpenPi0Config`: add `action_cond_enabled`, `action_cond_dropout_prob`, `refine_iters` fields.
  - `OpenPi0ForRLActionPrediction.__init__`: register `action_cond_in_proj` (zero-init) and wire config.
  - Override / subclass `embed_suffix`: accept `action_cond`, emit the extra token block with matching `att_masks`.
  - Update the four call sites that currently invoke `embed_suffix` to forward `action_cond`.
  - Add `sample_actions_refine` wrapper.
  - `default_forward` / `sft_forward`: apply CFG-style dropout on `action_cond` before calling `super().forward`.

- `rlinf/workers/rollout/hf/huggingface_worker.py`
  - Read offline-save config block in `__init__`.
  - In `generate()`, maintain `self.oar_buffer_list` parallel to `self.buffer_list`.
  - After `predict(...)` in each chunk step, construct and append an `OARStepResult`.
  - After the outer epoch loop, call `_save_oar_shards(global_step)`.

- `rlinf/data/io_struct.py`
  - Add `@dataclass OARStepResult`.
  - Add `@dataclass OARRolloutBuffer` with `append`, `to_dict`.

- `examples/embodiment/config/model/pi0.yaml` and `pi0_5.yaml`
  - Add the `openpi.action_cond` block with safe defaults.

- All existing run configs in `examples/embodiment/config/*.yaml` — no change required. `offline_save.enabled` defaults to `false` (opt-in only). `action_cond.enabled` defaults to `true`; zero-init of `action_cond_in_proj` guarantees bit-for-bit parity with baseline, so existing runs produce identical output with the feature available but dormant.

---

## 10. Appendix — chosen options at a glance

| Decision | Options | Chosen | Rationale |
|---|---|---|---|
| (a) Position of `a_cond` in suffix | A1: between state and action_time / A2: before state | **A1** | a_cond sees current state; symmetric with π0.5 (no state block) |
| (b) Init of `action_cond_in_proj` | B1: zero / B2: kaiming / B3: copy / B4: gated | **B1** | bit-for-bit fallback, standard ControlNet-style add-on pattern |
| (c) Attention direction | one-way / two-way | **one-way** | refinement lives in outer loop, not inside single forward |
| Purpose of (o,a,r) collector | 1 online / 2 offline / 3 both | **2 offline** | matches user intent; simplest to verify |
| Collector approach | α exporter / β parallel collector / γ worker group | **β** | decouples from PPO schema, avoids γ infra cost |
| D1 cadence | per chunk / per epoch / per global step | **per epoch** | off critical path, manageable file count |
| D2 layout | sharded .pt + metadata.json | **sharded .pt + metadata.json** | simple, robust, per-worker write (no coordination) |
| D3 fields | MDP core only (obs, action, reward, termination, task_ids, success_once) | **D3 default** | tokenizer-agnostic, no PPO-specific pollution |
| D4 config flag | `rollout.offline_save.*` off by default | **off by default** | zero impact on existing runs |
| E1 vs E2 (which action) | E1 executed / E2 both | **E1** | simpler; equals policy output when intervene is off |
| Granularity | chunk step / env sub-step | **chunk step** | honest MDP for chunk policy, no obs duplication |

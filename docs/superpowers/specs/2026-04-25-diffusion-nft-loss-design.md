# Design: DiffusionNFT-style Episode-wise Loss for the Action-Conditioned Expert

- **Date**: 2026-04-25
- **Author**: iterated with user (xiaokejiang2026@gmail.com)
- **Scope**: Add a new RL policy loss for the action expert that mirrors NVIDIA's DiffusionNFT (ICLR 2026) — forward-process flow-matching loss with positive/negative split driven by **episode-level** advantages — and integrate it with the action-conditioning port and offline (o, a, r) buffer landed in `2026-04-24`.
- **Builds on**: `docs/superpowers/specs/2026-04-24-action-conditioned-expert-and-oar-rollout-design.md`
- **Status**: awaiting user review (no code changes yet)

---

## 1. Motivation

The repo's current `compute_nft_actor_loss` (`rlinf/algorithms/losses.py:276`) is the **π-StepNFT** loss: chain-aware, step-wise, requires storing full denoising trajectories (`chains` + `denoise_inds` + `nft_xt`/`nft_v`/`nft_xnext`). It's a DPO-flavored loss over per-denoising-step transitions, and advantages are step-wise (per chunk-step GAE).

The user wants to swap to **DiffusionNFT** (NVlabs paper, ICLR 2026 Oral): forward-process loss that operates on **clean x0** noised at a freshly-sampled `t`, with positive/negative splits weighted by an **episode-level** reward signal. The action-conditioning port (`a_cond`) we just shipped feeds it cleanly, and the offline (o, a, r) buffer already contains everything the loss needs.

The two losses are fundamentally different:

| | Existing `nft-actor` (π-StepNFT) | DiffusionNFT (this spec) |
|---|---|---|
| Target | `x_next` from chain or `x0_target` | **Clean x0** |
| Reference models | `v_θ`, `v_old` | `v_θ`, `v_old`, **`v_ref`** |
| Pos/neg construction | Symmetric: `v_pos = v_old ± β·Δv_clipped` | Asymmetric: `pos = β·v_θ + (1−β)·v_old`, `neg = (1+β)·v_old − β·v_θ` |
| Weighting | Step-aware Gaussian variance `σ_i²` | **Adaptive** per-sample: `\|x0_pred − x0\|.mean(no-grad)` |
| Reduction | DPO-style: `softplus((β/2)·y·ΔE)` | Direct: `r·pos_loss + (1−r)·neg_loss` |
| Trajectory storage | Required (`chains`, `denoise_inds`, …) | **Not used** — fresh `t` per training step |
| Advantage granularity | Step-wise GAE | **Episode return**, per-task standardized, broadcast to chunks |
| KL anchor | `‖v_θ − v_old‖²` | `‖v_θ − v_ref‖²` (paper); `‖v_θ − v_old‖²` if `reference_mode="v_old"` (this spec's default) |

This spec is strictly additive. The existing `nft-actor` loss is untouched and remains selectable via `algorithm.policy_loss_name`.

---

## 2. Goals and non-goals

### Goals
- **G1**: Register a new policy loss `"diffusion-nft-actor"` in `rlinf/algorithms/losses.py`, faithful to the DiffusionNFT formulation (asymmetric pos/neg + adaptive weighting + KL anchor).
- **G2**: Compute advantages **per episode**, standardized **per task group** (Q4-ii), broadcast as a constant `r ∈ [0, 1]` to every chunk-step inside that episode.
- **G3**: Source `v_ref` cheaply: by default reuse `v_old` (zero memory cost); offer LoRA-disable mode (`reference_mode: "lora_disable"`) when `is_lora=true` for paper-faithful KL.
- **G4**: Reuse the action-conditioning `a_cond` port we already shipped — pass `clean_action.detach()` with 50% CFG dropout during the loss forward.
- **G5**: Use the **same fields** the OAR buffer captures (`observation/{image,wrist_image,state}`, `task_descriptions`, `executed_action`, `reward`, `done`, `terminations`, `truncations`, `task_ids`, `success_once`) as the loss's input data. In this iteration the actual data still flows over the existing in-memory `EmbodiedRolloutResult` wire to the actor (see §5.5) — these fields are already on the wire. The on-disk OAR shards remain an analysis artifact and can later become a Path-Y data source for offline training without re-instrumenting the rollout. The chain-based fields (`chains`, `denoise_inds`, `nft_xt`, …) are unused by this loss.
- **G6**: Default-off behavior: existing experiments (with `policy_loss_name: ppo-actor` or `nft-actor`) are unaffected.

### Non-goals
- No change to the action-expert architecture from the `2026-04-24` spec.
- No change to PPO/π-StepNFT loss code paths; both stay registered and runnable.
- No new rollout-collection logic; the OAR buffer (already shipped) is the data source.
- No `value_head` interaction. Under DiffusionNFT, the critic is unused; if `add_value_head=true` is set, its forward still runs (cheap) but its loss is zero-weighted in the new path.
- No frozen-clone reference (`Q2a` was explicitly dropped). This avoids ~4B params of duplicated state during full fine-tuning.
- No new evaluation metrics — wandb logs are reused (advantage stats, per-loss components).

---

## 3. Algorithm — episode-wise DiffusionNFT loss

### 3.1 Forward-process noising

For each training (chunk-step) sample:
- `x_0 ∈ ℝ^(B, H, action_dim)` — the executed action chunk (from OAR buffer).
- Sample `t ∼ U(ε, 1−ε)` with small `ε = 0.001` to avoid degenerate endpoints.
- Sample `noise ∼ N(0, I)` matching `x_0`'s shape.
- `x_t = (1 − t)·x_0 + t·noise`.

This is the standard rectified-flow / linear-flow forward process used by π0 / π0.5.

### 3.2 Three velocity forwards per training sample

| Symbol | Definition | Compute when? |
|---|---|---|
| `v_θ`   | Current policy: `f_θ(o, x_t, t, a_cond)` | **Always**, with grad |
| `v_old` | Rollout-time policy snapshot: `f_θ_old(o, x_t, t, a_cond)` | Always, **no grad** |
| `v_ref` | `v_old` if `reference_mode="v_old"`; or `f_θ(o, x_t, t, a_cond)` with `disable_adapter()` if `reference_mode="lora_disable"` | Always, **no grad** |

`v_old` is obtained via the standard mechanism Negfinetun-VLA already uses: weights synced at rollout time, frozen during the actor update window. The actor worker must hold a reference to those weights (already does — `_v_old` snapshot in PPO/NFT path is the same object).

### 3.3 Positive / negative prediction split

```
positive_pred = β · v_θ + (1 − β) · v_old.detach()
negative_pred = (1 + β) · v_old.detach() − β · v_θ
```

`β ∈ (0, 1]` is a small mixing coefficient (default `0.5`). `positive_pred` mixes `v_θ` toward `v_old`; `negative_pred` is the mirror image such that pulling `negative_pred → x_0` is equivalent to pulling `v_θ` *away* from being a good predictor.

### 3.4 Adaptive per-sample weighting

```
x0_pos = x_t − t · positive_pred
x0_neg = x_t − t · negative_pred

with torch.no_grad():
    w_pos = |x0_pos − x_0|.mean(over non-batch dims, keepdim).clip(min=1e-5)
    w_neg = |x0_neg − x_0|.mean(over non-batch dims, keepdim).clip(min=1e-5)

pos_loss = ((x0_pos − x_0)² / w_pos).mean(over non-batch dims)   # shape [B]
neg_loss = ((x0_neg − x_0)² / w_neg).mean(over non-batch dims)   # shape [B]
```

Adaptive weighting normalizes per-sample magnitude — empirically stabilizes training when reward sparsity makes `pos_loss` and `neg_loss` differ by orders of magnitude.

### 3.5 Reward → r mapping (Q4-ii — episode-wise, per-task)

Step-by-step on the rollout buffer (T = n_chunk_steps × rollout_epoch, B = batch dim):

1. **Walk the time-axis** to assign episode IDs `episode_id[t, b]` per chunk-step. A new episode begins when **either** `done[t-1, b]` **or** `truncations[t-1, b]` is true (or at t=0). Episodes are unique per-(b, run-of-non-terminated-chunks). Both `done` and `truncations` count as boundaries because a time-limited rollout still ends an MDP episode for credit-assignment purposes.
2. **Per-episode return**:
   ```
   R[k] = Σ reward[t, b, h]  for all (t, b, h) in episode k
   ```
   (Sum across the H sub-step rewards within each chunk too, since `reward[t, b]` is shape `[H]`.)
3. **Per-task group standardization**:
   ```
   for each task_g in unique(task_ids):
       group_episodes = episodes with task_ids == task_g
       μ_g, σ_g = mean(R[k] | k ∈ group_episodes), std(...) + ε
       adv[k] = (R[k] − μ_g) / σ_g           for k ∈ group_episodes
   ```
   When `task_ids` is missing or single-valued, this degrades to global standardization (one group). `ε = 1e-4`.
4. **Clip & map**:
   ```
   adv_clipped[k]    = clip(adv[k], −adv_clip_max, +adv_clip_max)
   adv_normalized[k] = adv_clipped[k] / adv_clip_max / 2 + 0.5     # ∈ [0, 1]
   r[k]              = clip(adv_normalized[k], 0, 1)
   ```
5. **Broadcast** `r[k]` to every chunk-step in episode k:
   ```
   r_per_step[t, b] = r[episode_id[t, b]]
   ```

`r_per_step` has shape `[T, B]` and is the per-sample weight in the loss.

### 3.6 Final loss

```
policy_loss_per_sample = (r · pos_loss + (1 − r) · neg_loss) / β    # shape [B]
policy_loss            = policy_loss_per_sample.mean() · adv_clip_max

kl_loss_per_sample = ((v_θ − v_ref)²).mean(over non-batch dims)     # shape [B]
kl_loss            = kl_loss_per_sample.mean()

total_loss = policy_loss + kl_beta · kl_loss
```

When `reference_mode="v_old"`, `v_ref ≡ v_old.detach()`, so `kl_loss = ((v_θ − v_old.detach())²).mean(...)`. This still serves as a trust-region anchor (drift penalty), just less faithful to the paper's reference KL semantics.

### 3.7 Action conditioning during the loss forward (Q3a)

For each of the three velocity forwards (`v_θ`, `v_old`, `v_ref`):

```
a_cond = _sample_action_cond(x_0)        # 50% zeros, 50% x_0.detach()
```

The `_sample_action_cond` helper from `OpenPi0ForRLActionPrediction` (already implemented per the `2026-04-24` spec, §3.5) is reused. Per-sample Bernoulli dropout means roughly half of the batch trains the cold-start path and half trains the conditioned path with the same single checkpoint.

**Subtle:** the dropout mask must be **identical** across the three forwards (`v_θ`, `v_old`, `v_ref`) — otherwise the KL penalty would penalize `v_θ` for using a different conditioning input than `v_ref`, not for diverging in policy space. The actor worker draws one mask per training step and passes the resulting `a_cond` tensor to all three forwards.

---

## 4. Reference handling (Q2cb)

```
algorithm.diffusion_nft.reference_mode ∈ {"v_old", "lora_disable"}
```

### 4.1 `v_old` mode (default)

`v_ref ≡ v_old.detach()`. No additional forward, no extra memory. The KL anchor becomes a trust-region penalty against the rollout-time policy — drift over training as `v_old` is refreshed each rollout. This is **not the paper's KL** but is cheap, stable, and matches what the existing π-StepNFT loss does.

### 4.2 `lora_disable` mode

When `is_lora=true` (config knob already exists at `actor.model.is_lora`), expose a context manager on the model:

```python
with model.disable_adapter():
    v_ref = model.get_velocity(...)  # frozen base SFT velocity
```

The `disable_adapter()` is provided by PEFT when the model has LoRA adapters; we add a thin wrapper on `OpenPi0ForRLActionPrediction` that delegates to whichever adapter library wraps the action expert (`peft`'s `LoraModel.disable_adapter()` or equivalent).

If `reference_mode="lora_disable"` is requested but `is_lora=false`, the actor worker raises a clear ValueError at startup — refuse to silently fall back.

### 4.3 Why no frozen clone (Q2a dropped)

Cloning the 4B-param model doubles memory of the action expert. Even with FSDP sharding, the per-rank ZeRO-3 state would grow proportionally. For LoRA runs, `lora_disable` is free. For full fine-tuning, `v_old` is acceptable for an initial implementation; we can add the clone option later if the KL anchor matters experimentally.

---

## 5. Implementation plan

### 5.1 New helper for episode advantages

New module `rlinf/algorithms/advantages_episode.py`:

```python
def compute_episode_returns_per_task(
    *,
    reward: torch.Tensor,         # [T, B, H] fp32
    done: torch.Tensor,           # [T, B, H] bool
    truncations: torch.Tensor,    # [T, B, H] bool
    task_ids: Optional[torch.Tensor],  # [T, B] int64 or None
    adv_clip_max: float,
    epsilon: float = 1e-4,
) -> tuple[torch.Tensor, dict]:
    """
    Return r_per_step ∈ [0,1] of shape [T, B], per Q4-ii pipeline.

    Algorithm: walk the time axis, assign episode IDs from done|truncations,
    sum rewards per episode, standardize per task group, clip, map to [0,1],
    broadcast back to the [T, B] grid.

    Returns (r_per_step, metrics_dict). Metrics include per-group n_episodes,
    R mean/std/min/max, fraction of episodes that ended via `done` vs `truncation`,
    and overall fraction of `r` near 0 / 1 (saturation indicator).
    """
```

Exact algorithm inline in §3.5. No allocation of intermediate `[T, B]` tensors larger than the input; episode walk is a Python loop over T (n_chunk_steps × rollout_epoch is small — typically <100), the per-episode aggregation is a `torch.scatter_add` over the resulting episode IDs.

### 5.2 New loss in `rlinf/algorithms/losses.py`

```python
@register_policy_loss("diffusion-nft-actor")
def compute_diffusion_nft_actor_loss(
    *,
    v_theta: torch.Tensor,           # [B, H, action_dim]
    v_old: torch.Tensor,             # [B, H, action_dim]
    v_ref: torch.Tensor,             # [B, H, action_dim]
    x_t: torch.Tensor,               # [B, H, action_dim]
    x_0: torch.Tensor,               # [B, H, action_dim]
    t: torch.Tensor,                 # [B]
    r_per_step: torch.Tensor,        # [B] (from advantages_episode helper, sliced for this minibatch)
    beta: float = 0.5,
    kl_beta: float = 1e-4,
    adv_clip_max: float = 1.0,
    **kwargs,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """DiffusionNFT episode-wise policy loss. See spec §3.6."""
```

Returns `(total_loss, metrics_dict)`. Metrics: `policy_loss`, `kl_loss`, `pos_loss`, `neg_loss`, `r_mean`, `r_sat_frac`, `pos_neg_ratio`.

### 5.3 Actor worker plumbing

In `rlinf/workers/actor/fsdp_actor_worker.py`, add a branch parallel to the existing PPO/NFT paths:

```python
# Pseudocode at the actor-training site
if cfg.algorithm.policy_loss_name == "diffusion-nft-actor":
    # 1. Aggregate advantages once per rollout batch (CPU, fast)
    r_per_step, adv_metrics = compute_episode_returns_per_task(
        reward=batch["reward"], done=batch["done"], truncations=batch["truncations"],
        task_ids=batch.get("task_ids"),
        adv_clip_max=cfg.algorithm.diffusion_nft.adv_clip_max,
        epsilon=cfg.algorithm.diffusion_nft.advantage.epsilon,
    )
    log_metrics(adv_metrics)

    # 2. Iterate minibatches over (T, B) flattened
    for mb in iter_minibatches(batch, r_per_step):
        x_0 = mb["executed_action"]                    # [b, H, action_dim]
        observation = build_observation(mb)            # multimodal obs
        t = sample_t(mb.batch_size)                    # [b], U(eps, 1-eps)
        noise = torch.randn_like(x_0)
        x_t = (1 - t.view(-1, 1, 1)) * x_0 + t.view(-1, 1, 1) * noise

        # Shared a_cond mask across the three forwards
        a_cond = self.model._sample_action_cond(x_0)

        # Three velocity forwards
        v_theta = self.model.get_velocity_for_oar(observation, x_t, t, a_cond=a_cond)
        with torch.no_grad():
            v_old = self.v_old_model.get_velocity_for_oar(observation, x_t, t, a_cond=a_cond)
            if cfg.algorithm.diffusion_nft.reference_mode == "lora_disable":
                with self.model.disable_adapter():
                    v_ref = self.model.get_velocity_for_oar(observation, x_t, t, a_cond=a_cond)
            else:
                v_ref = v_old   # alias; no second forward

        loss, metrics = compute_diffusion_nft_actor_loss(
            v_theta=v_theta, v_old=v_old, v_ref=v_ref,
            x_t=x_t, x_0=x_0, t=t,
            r_per_step=mb["r_per_step"],
            beta=cfg.algorithm.diffusion_nft.beta,
            kl_beta=cfg.algorithm.diffusion_nft.kl_beta,
            adv_clip_max=cfg.algorithm.diffusion_nft.adv_clip_max,
        )
        loss.backward()
        ...
```

### 5.4 New thin helper on the model

`OpenPi0ForRLActionPrediction.get_velocity_for_oar(observation, x_t, t, *, a_cond)`:

A single-forward convenience that runs `embed_prefix` + paligemma KV-cache + `get_velocity` (already plumbed for `action_cond`), returning velocity at the given `t`. Mostly a wrapper around existing methods so the actor worker isn't reaching into private internals. ~30 lines.

A second helper `disable_adapter()` returns a context manager that delegates to PEFT's adapter-disable when `is_lora=true`; raises with a clear message otherwise.

### 5.5 OAR buffer reader for actor consumption

The actor currently consumes `EmbodiedRolloutResult` shipped via the `actor_channel`. Two paths to make OAR data reach the actor:

- **Path X (preferred, smallest change)**: do NOT change the wire. The rollout worker continues to ship `EmbodiedRolloutResult` over `actor_channel`. The actor extracts the same fields (`forward_inputs["action"]`, `forward_inputs["observation/state"]`, `rewards`, `dones`, `terminations`, `truncations`, `task_ids`) — these are already in the existing wire. The DiffusionNFT loss path needs nothing the wire doesn't already carry.

- Path Y: have the actor read the on-disk OAR shards. Decoupled from the wire, but adds disk I/O + sync points. Defer.

Path X means: the offline OAR shards continue to be a side-effect for outside-of-training analysis; the in-memory PPO buffer feeds the new loss.

### 5.6 Files to touch

| File | Change |
|---|---|
| `rlinf/algorithms/losses.py` | Add `compute_diffusion_nft_actor_loss` registered as `"diffusion-nft-actor"`. |
| `rlinf/algorithms/advantages_episode.py` | **New file** — `compute_episode_returns_per_task`. |
| `rlinf/workers/actor/fsdp_actor_worker.py` | New branch for `policy_loss_name == "diffusion-nft-actor"`: episode advantages, fresh-t sampling, three forwards, shared `a_cond` mask. |
| `rlinf/models/embodiment/openpi/openpi_action_model.py` | Add `get_velocity_for_oar` helper + `disable_adapter()` context manager. |
| `examples/embodiment/config/libero_*_nft_actor_openpi*.yaml` | One new example config showing the new block. Existing configs untouched. |

---

## 6. Configuration

New nested block under `algorithm`:

```yaml
algorithm:
  policy_loss_name: diffusion-nft-actor    # one of: ppo-actor | nft-actor | diffusion-nft-actor
  diffusion_nft:
    beta: 0.5                              # mixing coefficient
    adv_clip_max: 1.0                      # advantage clip
    kl_beta: 1.0e-4                        # weight on ‖v_θ − v_ref‖²
    reference_mode: v_old                  # v_old (default) | lora_disable
    advantage:
      mode: per_task                       # per_task (Q4-ii default) | global
      epsilon: 1.0e-4                      # std-clip in standardization
```

Defaults are chosen to mirror the paper where direct (`beta=0.5`, `adv_clip_max=1.0`) and to be cheap / safe where the paper would otherwise need extra infrastructure (`reference_mode=v_old`, `kl_beta=1e-4` not `paper's beta=0.04`).

When `reference_mode: lora_disable` is set with `actor.model.is_lora: false`, the actor worker raises:

```
ValueError: diffusion_nft.reference_mode='lora_disable' requires actor.model.is_lora=true.
```

This is intentional — silent fallback to `v_old` would mask a config error.

---

## 7. End-to-end data flow

```
EnvWorker        RolloutWorker             ActorWorker (DiffusionNFT path)
─────────        ─────────────             ──────────────────────────────
                                          (consume EmbodiedRolloutResult batch)
                                            │
                                            ├──► compute_episode_returns_per_task
                                            │       walk time axis
                                            │       per-task standardization
                                            │       → r_per_step  [T, B] ∈ [0,1]
                                            │
                                            ├──► for each minibatch over (T, B):
                                            │       sample t ~ U(ε, 1-ε)
                                            │       x_t = (1−t)·x_0 + t·noise
                                            │       a_cond = _sample_action_cond(x_0)  ← shared mask
                                            │
                                            │       v_θ   = model(o, x_t, t, a_cond)               (grad)
                                            │       v_old = v_old_model(o, x_t, t, a_cond)         (no grad)
                                            │       v_ref = v_old (alias)
                                            │             | model_with_adapter_disabled(o, x_t, t, a_cond)  (no grad)
                                            │
                                            └──► compute_diffusion_nft_actor_loss
                                                    asymmetric pos/neg + adaptive weight
                                                    + KL anchor against v_ref
                                                    → loss.backward()
```

The OAR shards continue to write to disk per the `2026-04-24` spec — they're analysis artifacts, not the training input here.

---

## 8. Testing plan

### 8.1 Unit — episode aggregation
- **T1**: Synthetic `reward`, `done` patterns of known structure (e.g., 3 episodes of varying length) → episode IDs match expected; per-episode returns equal hand-computed sums.
- **T2**: Single-task batch (`task_ids` all equal) → per-task standardization equals global standardization.
- **T3**: Two task groups with very different reward scales → standardization neutralizes the scale; resulting `r` is centered ~0.5 in each group.
- **T4**: Pure-truncation episodes (no `done`, only `truncations` at end-of-rollout) → still treated as episode boundaries.
- **T5**: All rewards zero → `std → 0`; the `+ ε` floor prevents NaN; resulting `r` saturates to 0.5 across the board (no signal, no movement).

### 8.2 Unit — loss math
- **T6**: With `r=1` everywhere, gradient of `total_loss` wrt `v_θ` matches gradient of `pos_loss / β + kl_beta · kl` (no negative branch active).
- **T7**: With `r=0` everywhere, gradient matches gradient of `neg_loss / β + kl_beta · kl`.
- **T8**: With `v_θ == v_old == v_ref`, `kl_loss == 0` and `pos_loss == neg_loss` (because positive_pred == negative_pred when β cancels).
- **T9**: Adaptive weighting: feed a synthetic batch where `pos_loss` would be 100× `neg_loss` un-weighted; check that after weighting they're within 2×.

### 8.3 Integration — actor branch
- **T10**: `policy_loss_name="diffusion-nft-actor"` runs end-to-end on LIBERO-spatial for 5 training steps. Loss does not NaN; gradients on `action_cond_in_proj.weight` are nonzero (CFG dropout active); `v_θ`, `v_old`, `v_ref` shapes match.
- **T11**: With `reference_mode="lora_disable"` + `is_lora=true`, two-forward path runs; `v_ref` differs from `v_old` after a few updates (sanity).
- **T12**: With `reference_mode="lora_disable"` + `is_lora=false`, startup raises ValueError immediately.
- **T13**: With existing `policy_loss_name="nft-actor"`, behavior is byte-for-byte unchanged from before this spec (no regressions).

### 8.4 Backward-compat
- **T14**: A run with `policy_loss_name="ppo-actor"` and `diffusion_nft` config block absent works; no warnings, no crashes.

---

## 9. Rollout / migration plan

1. Land the helper module + new loss + actor branch behind `policy_loss_name="diffusion-nft-actor"`. Default configs untouched, so existing experiments keep running unchanged.
2. Run T1–T9 (unit tests) locally.
3. Add one example config: `examples/embodiment/config/libero_object_dnft_actor_openpi_pi05.yaml` mirroring an existing libero config but with the new policy loss + LoRA enabled (so we can exercise `lora_disable`). Run T10–T12 on a single GPU for a handful of steps.
4. Run a real LIBERO-spatial training (~few hundred steps) comparing reward curves to the existing `nft-actor` baseline. Sanity check, not a competitive eval.
5. If/when needed: add `reference_mode: frozen_clone` and per-prompt standardization (Q4-iii). Both are out of scope here.

No existing recipe is affected unless it explicitly opts in.

---

## 10. Open items / out of scope

- **Frozen-clone reference (Q2a)** — explicitly deferred. Add when full-FT runs need paper-faithful KL.
- **Per-step credit assignment hybrid** — staying with episode-wise per Q4 lock. If a hybrid is wanted later, it would mix the existing GAE pipeline with this loss; not in this spec.
- **Reward shaping per task** — not addressed. Q4-ii standardization is the only normalization applied; reward shaping (curriculum, success-only filtering) is orthogonal.
- **Variance-aware β scheduling** — DiffusionNFT paper uses a constant β; an annealing schedule could help but is unproven for VLA tasks. Defer.
- **Reader / offline dataloader for OAR shards** — still deferred per the `2026-04-24` spec.
- **Compatibility with `flow_grpo` / `dance` solvers at sampling time** — DiffusionNFT is solver-agnostic at sampling, so any of the existing `solver_type` values work for rollout. No changes required.
- **PPO value-head training during DiffusionNFT** — explicitly disabled; if `add_value_head=true` is set, the value head's parameters are excluded from the optimizer (or simply its loss term is zero-weighted). Decided at implementation time; either form is fine.

---

## 11. File-by-file change preview (for the implementation plan)

- `rlinf/algorithms/losses.py`
  - Add `compute_diffusion_nft_actor_loss` registered as `"diffusion-nft-actor"`.
  - Helper `_adaptive_weight(x_pred, x_target)` shared between pos/neg branches.

- `rlinf/algorithms/advantages_episode.py` (new)
  - `compute_episode_returns_per_task(...)` per §3.5.
  - Returns `(r_per_step, metrics_dict)`.

- `rlinf/workers/actor/fsdp_actor_worker.py`
  - New branch `if cfg.algorithm.policy_loss_name == "diffusion-nft-actor": ...`
  - Shares minibatch iteration and gradient accumulation with the existing PPO/NFT paths.
  - Produces per-step wandb logs + per-task advantage metrics.

- `rlinf/models/embodiment/openpi/openpi_action_model.py`
  - `get_velocity_for_oar(self, observation, x_t, t, *, a_cond)` — convenience helper combining `embed_prefix` + `get_velocity`.
  - `disable_adapter(self)` — context manager that delegates to PEFT's `disable_adapter()` if `is_lora=True`; raises otherwise.

- `examples/embodiment/config/libero_object_dnft_actor_openpi_pi05.yaml` (new)
  - Mirrors `libero_object_nft_actor_openpi_pi05.yaml` but with the new `policy_loss_name` and `diffusion_nft` block.

---

## 12. Appendix — chosen options at a glance

| Decision | Options considered | Chosen | Rationale |
|---|---|---|---|
| Q1 | a) add new / b) replace existing | **a) add new** | Existing π-StepNFT runs stay valid; selectable via config. |
| Q2 | a) frozen clone / b) lora-disable / c) reuse v_old | **c default + b opt-in (no a)** | Cheapest by default; faithful when LoRA is on. Frozen clone deferred. |
| Q3 | a) CFG dropout / b) v_old's x0_pred / c) always zero | **a) CFG dropout** | Already implemented; safe; trains the a_cond port. |
| Q4 | i) global / ii) per-task / iii) raw clip | **ii) per-task standardization, episode-wise** | Robust to multi-task reward scales; matches DiffusionNFT's stat-tracking spirit; falls back to global when single-task. |
| Q5 | reuse OAR / new wire | **reuse OAR (in-memory wire)** | No new collection; existing rollout payload already carries everything. |

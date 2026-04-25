# RL Post-training with Episode-wise DiffusionNFT

How to fine-tune the action-conditioned π0 / π0.5 expert with the DiffusionNFT-style loss landed in `2026-04-25`. This is the runbook that goes with the design spec at `docs/superpowers/specs/2026-04-25-diffusion-nft-loss-design.md`.

## What this loop does

- Roll out the current policy in a parallel simulator (LIBERO / ManiSkill / MetaWorld / CALVIN / RoboCasa / RoboTwin).
- Compute **episode-level returns** per task group, standardize them within each group, broadcast a per-episode `r ∈ [0, 1]` to every chunk-step.
- For each minibatch: sample a fresh `t ∼ U(ε, 1−ε)`, noise the executed action chunk to `x_t`, run three velocity forwards (`v_θ`, `v_old`, `v_ref`), apply the asymmetric pos/neg DiffusionNFT loss with adaptive weighting + KL anchor.
- Update θ on the action expert; the value head is unused.

The new `a_cond` port is exercised through CFG-style dropout (50% zeros, 50% `x_0.detach()`) — same dropout mask is shared across the three velocity forwards, so the KL anchor stays consistent.

## Prerequisites

1. A pre-trained checkpoint per `docs/pre-training.md` (Option A or B).
2. Simulator assets installed:
   ```bash
   hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./assets
   # LIBERO only:
   git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git path/to/LIBERO
   export LIBERO_REPO_PATH=path/to/LIBERO
   ```
3. `algorithm.kl_beta > 0` is required when using `reference_mode: v_old` so that the actor allocates `self.ref_model` at init (this is the same model the new loss reads as `v_old/v_ref`).
4. For `reference_mode: lora_disable`, set `actor.model.is_lora: true` and ensure the action expert is wrapped with PEFT LoRA (the helper `model.disable_adapter()` walks `paligemma_with_expert.paligemma`/`gemma_expert`/self looking for the disable hook).

## Configuration

Start from any existing LIBERO/ManiSkill NFT config and override the algorithm block. Example: clone `examples/embodiment/config/libero_spatial_nft_actor_openpi_pi05.yaml` to `libero_spatial_dnft_actor_openpi_pi05.yaml` and replace its `algorithm:` block:

```yaml
algorithm:
  # ─── select the new loss + advantage ────────────────────────────────
  adv_type: episode-task-norm        # episode return → per-task standardize → broadcast
  loss_type: diffusion-nft-actor     # the new DiffusionNFT loss

  # ─── DiffusionNFT-specific knobs ────────────────────────────────────
  diffusion_nft:
    beta: 0.5                        # mixing for pos/neg pred (paper default)
    adv_clip_max: 1.0
    kl_beta: 1.0e-4                  # weight on ‖v_θ − v_ref‖² anchor
    reference_mode: v_old            # v_old (default, free) | lora_disable (paper-faithful)

  # ─── must stay > 0 so the actor allocates self.ref_model at init ────
  kl_beta: 1.0e-4

  # ─── advantage clip max (loss reads this if diffusion_nft.adv_clip_max is unset) ─
  clip_ratio_high: 1.0

  # ─── leave existing knobs as in the cloned config ───────────────────
  rollout_epoch: 1
  group_size: 8
  reward_type: chunk_level
  # … (gamma, gae_lambda, etc. inherited; not used by this loss but harmless)
```

Verify the model block (in the model yaml inherited via `defaults`):

```yaml
# examples/embodiment/config/model/pi0_5.yaml — defaults already correct, listed for clarity
openpi:
  action_cond_enabled: true
  action_cond_dropout_prob: 0.5
  refine_iters: 0          # 0 during training; can bump to 1 at eval time
```

For LoRA mode:

```yaml
actor:
  model:
    is_lora: true
    lora_rank: 32

algorithm:
  diffusion_nft:
    reference_mode: lora_disable
```

## Run

Use the existing run script — it dispatches to `train_embodied_agent.py` with hydra overrides:

```bash
cd /vePFS/zundong/Negfinetun-VLA
bash examples/embodiment/run_embodiment.sh libero_spatial_dnft_actor_openpi_pi05 \
  actor.model.model_path=models/pi05_libero \
  runner.logger.experiment_name=libero_spatial_dnft_v1 \
  runner.max_epochs=300 \
  runner.save_interval=20 \
  runner.val_check_interval=20
```

ETA on a single H100, LIBERO-spatial, 64 parallel envs: ≈ 2.5 hours per 100 training steps (rollout + actor update). Multi-GPU FSDP scales near-linearly until the rollout simulator becomes the bottleneck.

## What to watch in wandb / tensorboard

The new loss writes these keys (all under `actor/`):

| key | meaning | healthy range |
|---|---|---|
| `actor/diffusion_nft_loss` | `policy_loss` (pre-KL) | drifts down over time |
| `actor/diffusion_nft_pos_loss` | `pos_loss.mean()` | comparable magnitude to neg_loss |
| `actor/diffusion_nft_neg_loss` | `neg_loss.mean()` | as above |
| `actor/diffusion_nft_kl_loss` | `‖v_θ − v_ref‖²` | starts ~0 (zero-init), grows slowly |
| `actor/diffusion_nft_total_loss` | full loss | dominated by policy in early training |
| `actor/diffusion_nft_r_mean` | mean of `r ∈ [0, 1]` | ≈ 0.5 (standardized) |
| `actor/diffusion_nft_r_sat_frac` | fraction of `r < 0.05` or `r > 0.95` | < 0.3 typical |
| `actor/diffusion_nft_pos_neg_ratio` | `pos_loss.mean / neg_loss.mean` | within ~0.5–2× |
| `actor/diffusion_nft_w_pos_mean` | adaptive weight for pos branch | gives a sense of error magnitude |
| `actor/diffusion_nft_w_neg_mean` | adaptive weight for neg branch | as above |

Plus the standard rollout metrics: `env/success_once_rate`, `env/episode_return`, etc.

## Eval / inference with refinement

The action-conditioning port supports an inference-time refinement loop (one cold pass with `a_cond=0` then `refine_iters` passes feeding the previous output back as `a_cond`). Set `refine_iters: 1` (or higher) when running eval to get the refined action chunk:

```bash
TIMESTAMP=YYYYMMDD-HHMMSS \
EXP_SUBPATH=libero_spatial_dnft_v1/checkpoints \
EVAL_NAME=dnft_eval_${TIMESTAMP} \
MIN_STEP=100 \
bash examples/embodiment/batch_eval_embodiment.sh libero_spatial_dnft_actor_openpi_pi05 \
  actor.model.openpi.refine_iters=1
```

`refine_iters > 0` doubles inference compute per chunk (cold pass + refinement); use it only when you want the refined action.

## Sanity checks

Before launching a long run, walk through these:

1. **Loss path actually fires.** Run for 1 step with `runner.max_epochs=1` and grep the log for `actor/diffusion_nft_loss`. If absent, `algorithm.loss_type` is wrong.
2. **`v_ref` is allocated.** With `reference_mode: v_old`, set `algorithm.kl_beta > 0` (any positive value triggers `self.ref_model` allocation in `fsdp_actor_worker.py:1005`). Without this, the worker raises a clear `ValueError` from inside the new loss branch.
3. **`disable_adapter` works.** With `reference_mode: lora_disable`, run a single forward and check that `v_old` differs numerically from `v_theta` (otherwise LoRA isn't actually loaded). If LoRA isn't wrapped, the worker raises a clear ValueError at startup.
4. **Episode IDs are sane.** `actor/diffusion_nft_r_mean ≈ 0.5` confirms standardization is working. If `r_mean` is stuck at 0 or 1, episode boundaries from `done | truncations` are likely missing — verify the env populates these fields.
5. **`a_cond` mask is shared.** The KL term should start near zero and grow gradually. A KL term that explodes immediately means the three forwards saw different `a_cond` masks (a regression in `_sample_action_cond` plumbing).

## Troubleshooting

| symptom | likely cause | fix |
|---|---|---|
| `ValueError: diffusion_nft.reference_mode='lora_disable' requires actor.model.is_lora=true` | LoRA not enabled | switch to `reference_mode: v_old` or set `is_lora: true` |
| `ValueError: ... self.ref_model` not allocated | `kl_beta=0` AND no `nft`-prefix loss | set `algorithm.kl_beta > 0` (any tiny value) |
| `actor/diffusion_nft_kl_loss` explodes from step 1 | `a_cond` mask not shared across the three forwards | check `_sample_action_cond` is called once per minibatch in the actor branch, not per forward |
| `r_mean` saturates near 0 or 1 | episode standardization degenerate (all rewards equal) | inspect `actor/diffusion_nft_r_sat_frac`; if > 0.9, the env reward is constant — fix the reward signal |
| OOM at start | three forwards × FSDP shards | set `algorithm.diffusion_nft.reference_mode: v_old` (one extra forward, not two), reduce `actor.micro_batch_size`, or enable gradient checkpointing |
| `data["action"]` shape mismatch | rollout produced a different action layout (e.g. raw policy output instead of post-intervene) | verify `huggingface_worker.update_intervene_actions` set `forward_inputs["action"]` to `[B, H*action_dim]` |

## When to stop

- The standard signal: `env/success_once_rate` plateaus on the eval suite for several save-intervals.
- DiffusionNFT-specific signal: `actor/diffusion_nft_pos_neg_ratio` settles between roughly 0.5 and 2.0. Values outside this range for many steps indicate the adaptive weighting is failing to balance the two branches — usually a sign the policy has collapsed (positive branch trivially small) or diverged (negative branch dominates).

## What to keep in the run artifacts

- The checkpoint directory at the end-of-training save (`global_step_*/actor/model.safetensors`).
- The wandb / tensorboard run.
- The hydra config dump (auto-saved by hydra under the run dir).
- Optionally: the offline OAR shards under `${runner.output_dir}/rollouts/global_step_*/rank_*.pt` if you set `rollout.offline_save.enabled: true`. These are useful for later analysis or for training a downstream model on the trajectories collected during this RL run.

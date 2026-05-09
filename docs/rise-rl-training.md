# RISE-style RL fine-tuning (action-conditioned VLA + WM rollouts + value bootstrap)

End-to-end imagined-rollout RL on top of the action-conditioned π0.5 VLA. Rollouts come from the OpenSora STDiT3 LIBERO-spatial world model; advantages are GAE-bootstrapped from a separately trained RISE progress value model; the actor is updated with the existing DiffusionNFT loss.

## What this path does

For each outer training step:

1. **Sync** actor weights → rollout worker.
2. **Rollout** (group-sampled, multi-action per starting obs):
   - For each starting state `s_0`, the rollout worker forks `group_size` independent action trajectories through the world model.
   - For every chunk-step the trained RISE value model scores `V(s_t)` and writes it into the rollout buffer's `prev_values` slot.
3. **GAE advantages** with `(r_t + γ·V(s_{t+1}) − V(s_t))`.
4. **DiffusionNFT actor loss** updates the action-conditioned VLA.

This produces the "(obs, action_i, reward_i)" group rollout pattern, with each sample also getting a `V(s)` from the same trained value model.

## Components

| Component | Source | Notes |
|---|---|---|
| Actor (init) | `logs/sft_libero_spatial_20260501-074344/.../global_step_30000/actor/model_state_dict/full_weights.pt` | joint-SFT action-conditioned π0.5 |
| World model | `models/opensora_libero_spatial/` | OpenSora STDiT3 + SDXL VAE + ResNet RM, libero-spatial trained |
| Value model | `RISE/policy_and_value/policy_offline_and_value/checkpoints/value_release_libero_spatial/value_release_libero_spatial/3000/` | RISE PI0Pytorch progress value, paligemma backbone, sigmoid head ∈ [0,1] |
| Loss | `rlinf/algorithms/losses.py:compute_diffusion_nft_actor_loss` | unchanged |
| Advantage | `rlinf/algorithms/advantages.py:compute_gae_advantages_and_returns` | unchanged |

## Configuration

`examples/embodiment/config/libero_spatial_rise_actor_openpi_pi05.yaml` (selectable knobs):

```yaml
runner:
  ckpt_path:        ".../full_weights.pt"   # joint-SFT init
  value_model_path: ".../3000"              # RISE value ckpt dir
  max_epochs:       300                     # outer steps; reduce for short runs
  save_interval:    20

algorithm:
  adv_type:        gae                      # value-bootstrap GAE
  loss_type:       diffusion-nft-actor
  group_size:      4                        # samples per starting obs (RISE-style)
  rollout_epoch:   8                        # outer-step rollout iterations
  gae_lambda:      0.95
  gamma:           0.99
```

## How to launch

```bash
bash logs/rise-train-launch.sh
```

The launcher's defaults run a 50-step, ~22 h training. Adjust `runner.max_epochs`, `algorithm.rollout_epoch`, and `env.train.total_num_envs` to retarget. Checkpoints land in `logs/rise-train-<timestamp>/.../checkpoints/global_step_<N>/`.

## Code that was wired

- `rlinf/models/value/rise_value_model.py` (new) — `RiseValueModel` wrapper. Loads `PI0Pytorch` from the in-tree RISE checkout via lazy `sys.path` injection, builds the inference Observation, returns `[B]` V(s) on CPU.
- `rlinf/workers/rollout/hf/huggingface_worker.py` — optional value-model load when `cfg.runner.value_model_path` is set; per-step V(s) override of `result["prev_values"]`.
- `rlinf/models/embodiment/openpi/openpi_action_model.py` — exposes `processed_obs` in `predict_action_batch` result; adds `truncate_to_chunk: bool = True` parameter to `get_velocity_for_oar` so the DiffusionNFT loss can request the full `action_horizon` output.
- `rlinf/workers/actor/fsdp_actor_worker.py`:
  - bug fix at the DiffusionNFT entrypoint: `data["forward_inputs"]["action"]` → `data["action"]` (the buffer's `to_dict` flattens `forward_inputs` to top-level keys);
  - `precision_processor` call between `input_transform` and `Observation.from_dict` so the loss observation lands on the actor's GPU;
  - infer the `H × action_dim` chunk shape from the buffered tensor (model-space, not env-space cfg fields);
  - pass `truncate_to_chunk=False` to all three `get_velocity_for_oar` calls so `v_theta`/`v_old`/`v_ref` match `x_t`/`x_0`;
  - `torch.cuda.synchronize()` + explicit FSDP `_training_state` reset before `sync_model_to_rollout` to work around a no_shard-mode race where `_unshard_params_ctx` asserts on `BACKWARD_POST != IDLE`.

## Dependency

Beyond the project's existing openpi venv, you need `kornia` (the value model's preprocessing module imports `kornia.augmentation` at import time):

```
/path/to/openpi/bin/python -m pip install kornia
```

All other `openpi_value` deps (flax, jax, augmax, orbax, …) are already present in the openpi venv at the same pinned versions.

## Cost reference

Per outer step, on 8× H20:

| `rollout_epoch` | wall per step | 50-step run |
|---|---|---|
| 2 | ~14 min | ~12 h |
| 4 | ~26 min | ~22 h |
| 8 (default) | ~50 min | ~42 h |

Rollout dominates (>95% of step time); WM forward (OpenSora STDiT3, 30-step denoise × 12 latent frames per WM step) is the bottleneck. Actor update + advantage compute is <30 s.

## Known issues

- The first step of training works but step 2 historically crashed in `_unshard_params_ctx` with `BACKWARD_POST != IDLE`. The explicit FSDP state reset in `sync_model_to_rollout` papers over this but relies on torch FSDP private API (`_handle._training_state`); if the torch version changes, that block needs to be revisited.
- The Ray dashboard ProxyError lines that appear at job exit are cosmetic; they fire on shutdown when the dashboard is unreachable through the cluster proxy and do not affect saved checkpoints.

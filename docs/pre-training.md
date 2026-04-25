# Pre-training the Action-Conditioned Expert

How to obtain a working π0 / π0.5 checkpoint that can serve as the starting point for action-conditioned RL fine-tuning. The action-conditioning port (`action_cond_in_proj`) is **zero-initialized** so this step does not strictly need a separate SFT phase — any existing RLinf SFT checkpoint loads cleanly into the modified architecture and produces bit-for-bit baseline output. A short warmup phase that exercises the new port via CFG-style dropout is recommended before kicking off RL.

## What gets trained here

- The PaliGemma vision-language tower and the Gemma-300M action expert from upstream openpi.
- The new `action_cond_in_proj: nn.Linear(action_dim, expert_width)` (zero-init, registered when `action_cond_enabled=true`).
- Other heads (`action_in_proj`, `action_out_proj`, optional `value_head`, optional `noise_head`).

## Prerequisites

### Environment

```bash
# host paths (per /root/.claude/CLAUDE.md)
export HF_HOME=~/shock/.CACHE/hf_cache
export XDG_CACHE_HOME=~/shock/.CACHE/xdg_cache
export UV_CACHE_DIR=~/shock/.CACHE/uv_cache
export PIP_CACHE_DIR=~/shock/.CACHE/pip_cache
export CONDA_ENVS_PATH=~/shock/conda_envs

# rlinf docker image (matches README)
docker run -it --rm --gpus all --shm-size 20g --network host \
  --name rlinf -v .:/workspace/RLinf \
  rlinf/rlinf:agentic-rlinf0.1-maniskill_libero
# inside the container:
source switch_env openpi
```

### Checkpoints

Use one of the RLinf-hosted SFT checkpoints as the pre-trained starting point. They already include `model.safetensors` + `physical-intelligence/<suite>/norm_stats.json` and load directly into this repo's `OpenPi0ForRLActionPrediction`.

```bash
source /vePFS/shock/toolbox/bin/unset-proxy
export HF_ENDPOINT=https://hf-mirror.com

# π0 LIBERO (≈ 4 B params)
/vePFS/shock/toolbox/bin/hfd RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT \
  --local-dir models/pi0_libero

# π0.5 LIBERO
/vePFS/shock/toolbox/bin/hfd RLinf/RLinf-Pi05-LIBERO-SFT \
  --local-dir models/pi05_libero
```

Other suites: `RLinf-Pi{0,05}-ManiSkill-25Main-SFT`, `-MetaWorld-SFT`, `-CALVIN-ABC-D-SFT`, `-RoboCasa`, `-Behavior`, `-RoboTwin-SFT-adjust_bottle`. See the org index at <https://huggingface.co/RLinf>.

### LIBERO assets (only if running LIBERO)

```bash
hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./assets
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git path/to/LIBERO
export LIBERO_REPO_PATH=path/to/LIBERO
```

## Configuration

Set `actor.model.model_path` to the downloaded checkpoint, and **leave the action-conditioning defaults on**:

```yaml
# examples/embodiment/config/model/pi0.yaml (or pi0_5.yaml) — verify these defaults
openpi:
  config_name: pi0_libero               # match suite to checkpoint
  action_cond_enabled: true             # zero-init port available; harmless if untouched
  action_cond_dropout_prob: 0.5         # CFG dropout for the warmup
  refine_iters: 0                       # one-pass during warmup
```

`action_cond_enabled: true` with zero-init means the model produces the same outputs as the unmodified upstream `PI0Pytorch` until the new projection accumulates non-zero weight. There is no risk of degrading the loaded SFT checkpoint.

## Option A — skip SFT warmup, go straight to RL

For most users this is the right choice. The DiffusionNFT RL post-training loop (see `docs/rl-post-training.md`) trains the action-conditioned port through CFG dropout while doing online RL. The first ~100–500 RL steps act as a de-facto warmup of `action_cond_in_proj`.

Validation that the SFT checkpoint loads cleanly into the modified arch:

```bash
python - <<'PY'
import torch
from omegaconf import OmegaConf
from rlinf.models import get_model

cfg = OmegaConf.create({
    "model_type": "openpi",
    "model_path": "models/pi0_libero",
    "action_dim": 7, "num_action_chunks": 10, "num_steps": 4, "use_proprio": True,
    "is_lora": False, "lora_rank": 32, "add_value_head": False, "precision": None,
    "openpi": {
        "config_name": "pi0_libero", "num_images_in_input": 2, "noise_level": 0.5,
        "action_chunk": 10, "num_steps": 4, "train_expert_only": True,
        "action_env_dim": 7, "solver_type": "flow_sde", "add_value_head": False,
        "detach_critic_input": False,
        "action_cond_enabled": True, "action_cond_dropout_prob": 0.5, "refine_iters": 0,
    },
})
m = get_model(cfg)
w = m.action_cond_in_proj.weight
print(f"action_cond_in_proj shape={tuple(w.shape)} sum={w.abs().sum().item():.6f}")
assert w.abs().sum().item() == 0.0, "expected zero-init"
print("[DONE] action_cond port present and zero-init")
PY
```

Expected output: `action_cond_in_proj shape=(1024, 7) sum=0.000000` and `[DONE] action_cond port present and zero-init`.

## Option B — short warmup via NFT loop with CFG dropout

If you want `action_cond_in_proj` to start RL with non-trivial weights, run a short loop using the existing π-StepNFT path (`loss_type: nft-actor`). My edit to `forward_nft` (in `rlinf/models/embodiment/openpi/openpi_action_model.py`) already applies CFG dropout on `action_cond` derived from `chains[:, -1]`, so the new projection sees gradients during this phase.

Pick any LIBERO config that uses NFT, then run for a short horizon:

```bash
cd /vePFS/zundong/Negfinetun-VLA
bash examples/embodiment/run_embodiment.sh libero_spatial_nft_actor_openpi_pi05 \
  runner.max_epochs=20 \
  runner.save_interval=10 \
  actor.model.model_path=models/pi05_libero
```

After ~10 saves you should see `action_cond_in_proj` weights move off zero. Verify:

```bash
python - <<'PY'
import safetensors.torch as st
sd = st.load_file("../results/.../checkpoints/global_step_10/actor/model.safetensors")
w = sd.get("action_cond_in_proj.weight")
print("action_cond_in_proj weight nonzero?", w.abs().sum().item() > 0)
PY
```

## Option C — pure SFT pre-training (advanced, custom)

The repo ships a generic `SFTRunner` (`rlinf/runners/sft_runner.py`) and `FSDPSftWorker` but **no example pipeline for the embodied action expert**. Building this would require:

1. A LIBERO/ManiSkill demonstration dataset reshaped to `(observation, x_0)` per chunk.
2. A `forward` path that calls the existing `OpenPi0ForRLActionPrediction.sft_forward` and the upstream openpi flow-matching loss.
3. A `train_sft_embodied.py` entrypoint mirroring `train_embodied_agent.py`.

This is out of scope for the action-conditioning project. Use Option A or B instead.

## What "pre-training done" looks like

- A checkpoint directory containing `model.safetensors` (with `action_cond_in_proj.weight` either zero or non-zero, depending on whether you ran Option A or B).
- The matching `physical-intelligence/<suite>/norm_stats.json` file (carries through automatically when you start from an RLinf SFT checkpoint).
- Optionally, a brief sanity check that `model.predict_action_batch(env_obs)` produces sensible action chunks for a few test observations.

After this, proceed to `docs/rl-post-training.md`.

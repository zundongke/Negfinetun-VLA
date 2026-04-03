# $\pi$-StepNFT: Wider Space Needs Finer Steps in Online RL for Flow-based VLAs

**Main Results** (see paper for full protocol)
- **LIBERO (few-shot):** $\pi$-StepNFT improves performance by **32.9%** over SFT.
- **ManiSkill (OOD generalization):** $\pi$-StepNFT improves OOD success by **11.1%** over critic-based baselines by mitigating multimodal overfitting.

## Quick Start

For environment setup and simulator configuration details, please refer to the [RLinf](https://github.com/RLinf/RLinf) repository.

### Installation

Run experiments using the Docker image.

```bash
docker run -it --rm --gpus all \
   --shm-size 20g \
   --network host \
   --name rlinf \
   -v .:/workspace/RLinf \
   rlinf/rlinf:agentic-rlinf0.1-maniskill_libero
   # For faster mirror downloads in mainland China, you can use:
   # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.1-maniskill_libero
```

Switch to the corresponding virtual environment using the built-in `switch_env` tool:

```bash
source switch_env openpi
```

For Maniskill,
```
cd <path_to_pi_StepNFT>/rlinf/envs/maniskill
# For faster downloads in mainland China, you can set:
# export HF_ENDPOINT=https://hf-mirror.com
hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./assets
```

### Training

```bash
bash examples/embodiment/run_embodiment.sh libero_object_nft_actor_openpi
```

### Evaluation
```bash
# Batch eval on embodiment checkpoints (auto-scan global_step_* in descending order)
TIMESTAMP=YOUR_TIMESTAMP \
EXP_SUBPATH=maniskill_nft_actor_openpi/checkpoints \
EVAL_NAME=embodiment_${TIMESTAMP} \
MIN_STEP=160 \
bash examples/embodiment/batch_eval_embodiment.sh maniskill_ppo_openvlaoft
```

```bash
# Batch eval on ManiSkill OOD tasks across multiple envs
TIMESTAMP=YOUR_TIMESTAMP \
EXP_SUBPATH=maniskill_nft_actor_openpi/checkpoints \
CONFIG_NAME=YOUR_CFG_NAME \
EVAL_NAME=mani_ood_${TIMESTAMP} \
MIN_STEP=160 \
bash examples/embodiment/batch_eval_mani_ood.sh
```






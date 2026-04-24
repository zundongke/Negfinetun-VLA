# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import gc
import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from rlinf.config import SupportedModel
from rlinf.data.io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    OARRolloutBuffer,
    OARStepResult,
)
from rlinf.models import get_model
from rlinf.scheduler import Channel, Cluster, CollectiveGroupOptions, Worker
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.rollout.hf.utils import init_real_obs


class MultiStepRolloutWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.should_stop = False

        self.actor_group_name = cfg.actor.group_name
        self.device = torch.cuda.current_device()

        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.rollout.get("enable_offload", False)

        self.placement = HybridComponentPlacement(cfg, Cluster())

        actor_world_size = self.placement.get_world_size("actor")
        self.actor_weight_src_rank = self._rank % actor_world_size

        # Sync weight comm options
        max_ctas = cfg.rollout.get("sync_weight_nccl_max_ctas", None)
        min_ctas = cfg.rollout.get("sync_weight_nccl_min_ctas", None)
        self._sync_weight_comm_options = CollectiveGroupOptions(
            accel_max_ctas=max_ctas, accel_min_ctas=min_ctas
        )

        # Offline (o, a, r) collection config (see spec 2026-04-24 §4.8)
        offline_cfg = cfg.rollout.get("offline_save", None)
        self.offline_save_enabled = bool(
            offline_cfg.get("enabled", False) if offline_cfg is not None else False
        )
        self.offline_save_dir = (
            str(offline_cfg.get("output_dir", "./data/rollouts"))
            if offline_cfg is not None
            else "./data/rollouts"
        )
        self.offline_save_start_step = int(
            offline_cfg.get("start_step", 0) if offline_cfg is not None else 0
        )
        self._global_step = 0

    def init_worker(self):
        rollout_model_config = copy.deepcopy(self.cfg.actor.model)
        with open_dict(rollout_model_config):
            rollout_model_config.precision = self.cfg.rollout.model.precision
            rollout_model_config.model_path = self.cfg.rollout.model.model_path

        self.hf_model = get_model(rollout_model_config)

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            self.hf_model.load_state_dict(model_dict)

        self.hf_model.eval()

        self.setup_sample_params()
        if self.enable_offload:
            self.offload_model()

    def setup_sample_params(self):
        # length parameters for rollout
        self._length_params = OmegaConf.to_container(
            self.cfg.algorithm.length_params, resolve=True
        )
        # sampling parameters for rollout
        self._sampling_params = OmegaConf.to_container(
            self.cfg.algorithm.sampling_params, resolve=True
        )
        self._train_sampling_params = {
            "do_sample": self._sampling_params["do_sample"],
            "temperature": self._sampling_params["temperature_train"]
            if self._sampling_params["do_sample"]
            else 1.0,
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

        self._eval_sampling_params = {
            "do_sample": True
            if self._sampling_params.get("temperature_eval", -1) > 0
            else False,
            "temperature": self._sampling_params["temperature_eval"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

    def predict(self, env_obs, mode="train", task_ids=None):
        kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.MLP_POLICY,
            SupportedModel.GR00T,
            SupportedModel.CNN_POLICY,
        ]:
            kwargs = {"mode": mode}

        kwargs["return_obs"] = not hasattr(self.hf_model, "q_head")

        with torch.no_grad():
            actions, result = self.hf_model.predict_action_batch(
                env_obs=env_obs,
                **kwargs,
            )
            if task_ids is not None:
                forward_inputs = result.get("forward_inputs", {})
                forward_inputs["task_ids"] = torch.as_tensor(task_ids)
                result["forward_inputs"] = forward_inputs

        return actions, result

    def get_dones_and_rewards(
        self, env_output: dict[str, torch.Tensor], extracted_obs: dict[str, Any]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, Any] | None]:
        """
        Get dones and rewards from environment batch, handling auto_reset if needed.

        Args:
            env_output: Environment batch containing dones, rewards, and optionally final_obs

        Returns:
            Tuple of (dones, rewards, real_extracted_obs). dones and rewards are tensors.
        """
        # First step: no rewards yet, only dones
        real_extracted_obs = None
        if env_output["rewards"] is None:
            if hasattr(self.hf_model, "q_head"):
                real_extracted_obs = init_real_obs(extracted_obs)
            return (
                env_output["dones"].bool().cpu().contiguous(),
                None,
                real_extracted_obs,
            )

        dones = env_output["dones"].bool().cpu().contiguous()
        rewards = env_output["rewards"].cpu().contiguous()

        # Handle auto_reset: add bootstrap value to rewards for done episodes
        # Note: currently this is not correct for chunk-size>1 with partial reset
        if dones.any() and self.cfg.env.train.auto_reset:
            if hasattr(self.hf_model, "value_head") or hasattr(self.hf_model, "q_head"):
                final_obs = env_output["final_obs"]
                with torch.no_grad():
                    final_extracted_obs = self.hf_model.preprocess_env_obs(final_obs)
                    if hasattr(self.hf_model, "q_head"):
                        real_extracted_obs = init_real_obs(final_extracted_obs)
                    actions, result = self.predict(
                        final_extracted_obs, task_ids=env_output.get("task_ids")
                    )
                    if "prev_values" in result:
                        _final_values = result["prev_values"]
                    else:
                        _final_values = torch.zeros_like(actions[:, 0])
                final_values = torch.zeros_like(_final_values[:, 0])  # [bsz, ]
                last_step_dones = dones[:, -1]  # [bsz, ]

                final_values[last_step_dones] = _final_values[:, 0][last_step_dones]

                # Add bootstrap value to the last step of done episodes
                rewards[:, -1] += self.cfg.algorithm.gamma * final_values.cpu()

        if real_extracted_obs is None and hasattr(self.hf_model, "q_head"):
            real_extracted_obs = init_real_obs(extracted_obs)
        return dones, rewards, real_extracted_obs

    async def sync_model_from_actor(self):
        """Sync model parameters from the actor worker."""
        param_state_dict = await self.recv(
            self.actor_group_name,
            src_rank=self.actor_weight_src_rank,
            async_op=True,
            options=self._sync_weight_comm_options,
        ).async_wait()

        self.hf_model.load_state_dict(param_state_dict)
        del param_state_dict
        gc.collect()
        torch.cuda.empty_cache()

    def update_intervene_actions(self, env_output, forward_inputs):
        intervene_actions = env_output["intervene_actions"]
        intervene_flags = env_output["intervene_flags"]
        if intervene_actions is not None:
            if "action" in forward_inputs:
                policy_action = forward_inputs["action"].to(intervene_actions.device)
                policy_action = policy_action.reshape(
                    policy_action.shape[0], self.hf_model.num_action_chunks, -1
                )
                intervene_actions = intervene_actions.reshape(
                    intervene_actions.shape[0], self.hf_model.num_action_chunks, -1
                )
                action = intervene_actions * intervene_flags[
                    ..., None
                ] + policy_action * (~intervene_flags[..., None])
                action = action.reshape(action.shape[0], -1)
                forward_inputs["action"] = action
            else:
                raise NotImplementedError(f"{forward_inputs.keys()=}")
        return forward_inputs

    def _append_oar_step(
        self,
        *,
        stage_id: int,
        env_output: dict,
        extracted_obs: dict,
        actions: Any,
        rewards: Any,
        dones: Any,
    ) -> None:
        """Build an OARStepResult from the current chunk step and append to the per-stage buffer.

        Uses raw env_output.obs (images, state, task_descriptions) to stay
        tokenizer-agnostic. See spec 2026-04-24 §4.3.
        """
        obs = env_output["obs"]

        main_images = obs.get("main_images")
        if main_images is not None and not torch.is_tensor(main_images):
            main_images = torch.as_tensor(main_images)

        wrist_images = obs.get("wrist_images")
        if wrist_images is not None and not torch.is_tensor(wrist_images):
            wrist_images = torch.as_tensor(wrist_images)

        state = obs.get("states")
        if state is not None and not torch.is_tensor(state):
            state = torch.as_tensor(state)

        task_descriptions = obs.get("task_descriptions")
        task_descriptions = list(task_descriptions) if task_descriptions is not None else []

        # actions comes from predict() — possibly numpy, possibly [B, H*ad] or [B, H, ad]
        act = actions
        if not torch.is_tensor(act):
            act = torch.as_tensor(act)
        H = int(self.cfg.actor.model.num_action_chunks)
        action_dim = int(self.cfg.actor.model.action_dim)
        if act.dim() == 2 and act.shape[1] == H * action_dim:
            act = act.reshape(act.shape[0], H, action_dim)
        act = act.to(dtype=torch.float32)

        # First reset step has rewards=None; represent as zeros so stacking works.
        if rewards is None:
            bsz = state.shape[0] if state is not None else act.shape[0]
            r = torch.zeros((bsz, H), dtype=torch.float32)
        else:
            r = rewards if torch.is_tensor(rewards) else torch.as_tensor(rewards)
            r = r.to(dtype=torch.float32)

        step = OARStepResult(
            main_images=main_images,
            wrist_images=wrist_images,
            state=state,
            task_descriptions=task_descriptions,
            executed_action=act,
            reward=r,
            done=dones,
            terminations=env_output.get("terminations"),
            truncations=env_output.get("truncations"),
            task_ids=env_output.get("task_ids"),
            success_once=env_output.get("success_once"),
        )
        self.oar_buffer_list[stage_id].append(step)

    async def generate(
        self, input_channel: Channel, output_channel: Channel, actor_channel: Channel
    ):
        if self.enable_offload:
            self.reload_model()

        self.buffer_list = [
            EmbodiedRolloutResult(rollout_epoch=self.cfg.algorithm.rollout_epoch)
            for _ in range(self.num_pipeline_stages)
        ]

        # Offline (o, a, r) collection runs alongside the PPO/NFT buffer.
        # See spec 2026-04-24 §4.2.
        if self.offline_save_enabled:
            self.oar_buffer_list = [
                OARRolloutBuffer() for _ in range(self.num_pipeline_stages)
            ]
        else:
            self.oar_buffer_list = []

        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        for _ in tqdm(
            range(self.cfg.algorithm.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            last_extracted_obs = [None for i in range(self.num_pipeline_stages)]
            last_forward_inputs = [
                None for i in range(self.num_pipeline_stages)
            ]  # save actions

            for _ in range(n_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel)

                    if last_forward_inputs[stage_id] is not None:
                        last_forward_inputs[stage_id] = self.update_intervene_actions(
                            env_output, last_forward_inputs[stage_id]
                        )

                    extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                    dones, rewards, real_extracted_obs = self.get_dones_and_rewards(
                        env_output, extracted_obs
                    )
                    actions, result = self.predict(
                        extracted_obs, task_ids=env_output.get("task_ids")
                    )
                    success_once = env_output.get("success_once")
                    if success_once is not None and torch.is_tensor(success_once):
                        if torch.any(success_once):
                            if not hasattr(self, "_logged_success_once_rollout"):
                                print(
                                    f"[rollout_worker] success_once nonzero at rollout stage, "
                                    f"count={int(success_once.to(torch.int).sum().item())}",
                                    flush=True,
                                )
                                self._logged_success_once_rollout = True
                    chunk_step_result = ChunkStepResult(
                        prev_logprobs=result["prev_logprobs"],
                        prev_values=result["prev_values"],
                        dones=dones,
                        truncations=env_output["truncations"],
                        terminations=env_output["terminations"],
                        rewards=rewards,  # the first step is reset step, reward is none, which will not be appended to the buffer
                        success_once=success_once,
                        forward_inputs=last_forward_inputs[stage_id],
                    )
                    self.buffer_list[stage_id].append_result(chunk_step_result)
                    if last_extracted_obs[stage_id] is not None and hasattr(
                        self.hf_model, "q_head"
                    ):
                        self.buffer_list[stage_id].add_transition(
                            last_extracted_obs[stage_id], real_extracted_obs
                        )

                    # Atomic (o, a, r) capture for offline persistence (spec §4).
                    if self.offline_save_enabled:
                        self._append_oar_step(
                            stage_id=stage_id,
                            env_output=env_output,
                            extracted_obs=extracted_obs,
                            actions=actions,
                            rewards=rewards,
                            dones=dones,
                        )

                    last_extracted_obs[stage_id] = extracted_obs
                    last_forward_inputs[stage_id] = result["forward_inputs"]

                    self.send_chunk_actions(output_channel, actions)

            for stage_id in range(self.num_pipeline_stages):
                env_output = await self.recv_env_output(input_channel)
                last_forward_inputs[stage_id] = self.update_intervene_actions(
                    env_output, last_forward_inputs[stage_id]
                )

                extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                # Get dones and rewards from environment batch (final step of epoch)
                dones, rewards, real_extracted_obs = self.get_dones_and_rewards(
                    env_output, extracted_obs
                )
                self.buffer_list[stage_id].dones.append(dones)
                self.buffer_list[stage_id].truncations.append(env_output["truncations"])
                self.buffer_list[stage_id].terminations.append(
                    env_output["terminations"]
                )
                self.buffer_list[stage_id].rewards.append(rewards)
                if env_output.get("success_once") is not None:
                    self.buffer_list[stage_id].success_once.append(
                        env_output["success_once"].cpu().contiguous()
                    )
                self.buffer_list[stage_id].forward_inputs.append(
                    put_tensor_device(last_forward_inputs[stage_id], "cpu")
                )

                with self.worker_timer():
                    actions, result = self.predict(
                        extracted_obs, task_ids=env_output.get("task_ids")
                    )
                # For the final step, we only need prev_values for bootstrapping
                # This is a special case that doesn't create a full ChunkStepResult
                if "prev_values" in result:
                    self.buffer_list[stage_id].prev_values.append(
                        result["prev_values"].cpu().contiguous()
                    )
                if hasattr(self.hf_model, "q_head"):
                    self.buffer_list[stage_id].add_transition(
                        last_extracted_obs[stage_id], real_extracted_obs
                    )

        for i in range(self.num_pipeline_stages):
            self.send_rollout_batch(actor_channel, i)

        # Offline (o, a, r) shards written after the in-memory PPO buffer has
        # been handed off to the actor. See spec 2026-04-24 §4.5.
        self._save_oar_shards()

        if self.enable_offload:
            self.offload_model()

    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        if self.enable_offload:
            self.reload_model()

        n_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        for _ in tqdm(
            range(self.cfg.algorithm.eval_rollout_epoch),
            desc="Evaluating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            for _ in range(n_chunk_steps):
                for _ in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel, mode="eval")
                    extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                    actions, _ = self.predict(
                        extracted_obs, mode="eval", task_ids=env_output.get("task_ids")
                    )
                    self.send_chunk_actions(output_channel, actions, mode="eval")

        if self.enable_offload:
            self.offload_model()

    def offload_model(self):
        self.hf_model = self.hf_model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def reload_model(self):
        self.hf_model = self.hf_model.to(self.device)

    async def recv_env_output(
        self, input_channel: Channel, mode="train"
    ) -> dict[str, torch.Tensor]:
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        # Use asyncio so that it can run alongside async weight syncing
        env_output = await input_channel.get(
            key=f"{self._rank}_{mode}", async_op=True
        ).async_wait()
        return env_output

    def send_chunk_actions(self, output_channel: Channel, chunk_actions, mode="train"):
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        output_channel.put(
            item=chunk_actions, key=f"{self._rank}_{mode}", async_op=True
        )

    def send_rollout_batch(self, actor_channel: Channel, stage_id: int):
        # send rollout_batch to actor
        split_num = self.get_actor_split_num()
        splitted_rollout_result = self.buffer_list[stage_id].to_splitted_dict(split_num)
        for i in range(split_num):
            actor_channel.put(item=splitted_rollout_result[i], async_op=True)

    def get_actor_split_num(self):
        send_num = self.placement.get_world_size("rollout") * self.num_pipeline_stages
        recv_num = self.placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        return split_num

    def set_global_step(self, global_step):
        self._global_step = int(global_step)
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)

    def _save_oar_shards(self) -> None:
        """Persist the per-stage OAR buffers to disk, one shard per (rank, stage).

        One shard per rollout-rank per pipeline stage under
        ``{output_dir}/global_step_{N}/``. Rank 0 additionally writes a small
        ``metadata.json`` describing shapes and dtypes. Never raises — disk
        errors are logged and swallowed to keep the training loop alive.
        """
        if not self.offline_save_enabled:
            return
        if self._global_step < self.offline_save_start_step:
            return
        if not hasattr(self, "oar_buffer_list") or not self.oar_buffer_list:
            return

        out_dir = Path(self.offline_save_dir) / f"global_step_{self._global_step}"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"[rollout_worker] offline_save: mkdir {out_dir} failed: {e}", flush=True)
            return

        example_dict = None
        for stage_id, buf in enumerate(self.oar_buffer_list):
            shard_path = out_dir / f"rank_{self._rank}_stage_{stage_id}.pt"
            try:
                payload = buf.to_dict()
                torch.save(payload, shard_path)
                if example_dict is None:
                    example_dict = payload
            except (OSError, RuntimeError) as e:
                print(
                    f"[rollout_worker] offline_save: write {shard_path} failed: {e}",
                    flush=True,
                )

        if self._rank == 0 and example_dict is not None:
            try:
                meta = {
                    "global_step": self._global_step,
                    "world_size": self.placement.get_world_size("rollout"),
                    "pipeline_stage_num": self.num_pipeline_stages,
                    "action_dim": int(self.cfg.actor.model.action_dim),
                    "num_action_chunks": int(self.cfg.actor.model.num_action_chunks),
                    "shapes": {
                        k: (
                            list(v.shape) if torch.is_tensor(v) else None
                        )
                        for k, v in example_dict.items()
                    },
                    "dtypes": {
                        k: str(v.dtype) if torch.is_tensor(v) else None
                        for k, v in example_dict.items()
                    },
                }
                with (out_dir / "metadata.json").open("w") as f:
                    json.dump(meta, f, indent=2)
            except OSError as e:
                print(
                    f"[rollout_worker] offline_save: metadata.json failed: {e}",
                    flush=True,
                )

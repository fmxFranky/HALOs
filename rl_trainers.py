import contextlib
import functools
import gc
import json
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import transformers
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.api import FullOptimStateDictConfig, FullStateDictConfig
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from transformers import AutoTokenizer

import dataloader
from models import AutoModelForCausalLMWithValueHead
from utils import (
    all_gather_if_needed,
    delete_dict,
    entropy_from_logits,
    formatted_dict,
    get_batch_logps,
    get_block_class_from_model,
    masked_mean,
    masked_var,
    pad_to_length,
    rank0_print,
    rowwise_product,
    slice_and_move_batch_for_device,
)

torch.backends.cuda.matmul.allow_tf32 = True


class RewardModelTrainer(object):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        config: DictConfig,
        train_iterator: dataloader.DataLoader,
        eval_iterator: dataloader.DataLoader,
        policy: nn.Module,
        reference_model: Optional[nn.Module] = None,
        rank: int = 0,
        world_size: int = 1,
        fsdp: bool = False,
    ):
        """A trainer for a language model, supporting either SFT, HALO, or offline PPO training."""
        self.seed = config.seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.rank = rank
        self.device = torch.device("cuda", self.rank)
        self.world_size = world_size
        self.config = config
        self.run_dir = config.local_run_dir
        self.fsdp = fsdp

        self.tokenizer = tokenizer
        self.policy = policy
        self.policy_dtype = getattr(torch, config.model.policy_dtype)
        self.reference_model = reference_model
        self.example_counter = 0
        self.batch_counter = 0

        self.train_iterator = train_iterator
        self.eval_iterator = eval_iterator
        self.eval_batches = list(self.eval_iterator)
        rank0_print(
            f"Loaded {len(self.eval_batches)} eval batches of size {config.model.eval_batch_size}"
        )

        if self.fsdp:
            self.shard()

        self.is_mistral = "mistral" in self.config.model.name_or_path.lower()

    def shard(self):
        """
        Shard the policy model and reference model (if applicable) using FDSP.
        """
        assert (
            self.config.model.block_name is not None
        ), "must specify model.block_name (e.g., GPT2Block or GPTNeoXLayer) for FSDP"
        wrap_class = get_block_class_from_model(
            self.policy.pretrained_model,
            self.config.model.block_name,
        )
        model_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={wrap_class},
        )

        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=self.rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False,
        )

        rank0_print("Sharding RM ...")
        mp_dtype = (
            getattr(torch, self.config.model.fsdp_policy_mp)
            if self.config.model.fsdp_policy_mp is not None
            else None
        )
        policy_mp_policy = MixedPrecision(
            param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype
        )

        self.policy.pretrained_model = FSDP(
            self.policy.pretrained_model,
            **shared_fsdp_kwargs,
            mixed_precision=policy_mp_policy,
        )

        # shard the value head according to size
        v_head_shared_fsdp_kwargs = dict(
            auto_wrap_policy=functools.partial(
                size_based_auto_wrap_policy, min_num_params=100
            ),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=self.rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False,
        )
        self.policy.v_head = FSDP(self.policy.v_head, **v_head_shared_fsdp_kwargs)

        if self.config.model.activation_checkpointing:
            rank0_print("Attempting to enable activation checkpointing...")
            try:
                # use activation checkpointing, according to:
                # https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/
                # first, verify we have FSDP activation support ready by importing:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    checkpoint_wrapper,
                    apply_activation_checkpointing,
                    CheckpointImpl,
                )
            except Exception as e:
                rank0_print("FSDP activation checkpointing not available:", e)
            else:
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print("Applying activation checkpointing wrapper to policy...")

                apply_activation_checkpointing(
                    self.policy.pretrained_model,
                    checkpoint_wrapper_fn=checkpoint_wrapper,
                    check_fn=check_fn,
                )

                rank0_print("FSDP activation checkpointing enabled!")

        print("Loaded RM on rank", self.rank)
        dist.barrier()

    def concatenated_inputs(
        self, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor. The first half is chosen outputs, the second half is rejected.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        max_length = max(
            batch["chosen_combined_input_ids"].shape[1],
            batch["rejected_combined_input_ids"].shape[1],
        )
        concatenated_batch = {}
        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                pad_value = -100 if "labels" in k else 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(
                    batch[k], max_length, pad_value=pad_value
                )
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                pad_value = -100 if "labels" in k else 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                )
        return concatenated_batch

    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        def ctx():
            return (
                FSDP.summon_full_params(self.policy, writeback=False, recurse=False)
                if self.fsdp
                else contextlib.nullcontext()
            )

        with ctx():
            policy_output = self.policy.generate(
                batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.config.model.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                top_p=self.config.top_p,
            )

            policy_output = pad_to_length(
                policy_output, self.config.model.max_length, self.tokenizer.pad_token_id
            )
            policy_output = all_gather_if_needed(
                policy_output, self.rank, self.world_size
            )
            policy_output_decoded = self.tokenizer.batch_decode(
                policy_output, skip_special_tokens=True
            )

        return policy_output_decoded

    def forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[
        Dict[str, torch.LongTensor],
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        concatenated_batch = self.concatenated_inputs(batch)
        scores, end_scores = model(
            concatenated_batch["concatenated_combined_input_ids"],
            attention_mask=concatenated_batch["concatenated_combined_attention_mask"],
            use_cache=(not self.is_mistral),
        )
        scores = scores.to(self.policy_dtype)
        end_scores = end_scores.to(self.policy_dtype)
        higher_rewards, lower_rewards = scores.squeeze(dim=-1).chunk(chunks=2, dim=0)
        # size = (B,)
        higher_end_reward, lower_end_reward = end_scores.squeeze(dim=-1).chunk(
            chunks=2, dim=0
        )
        return (
            concatenated_batch,
            higher_rewards,
            lower_rewards,
            higher_end_reward,
            lower_end_reward,
        )

    def loss(
        self,
        concatenated_batch: Dict[str, torch.LongTensor],
        higher_rewards: torch.FloatTensor,
        lower_rewards: torch.FloatTensor,
        higher_end_reward: torch.FloatTensor,
        lower_end_reward: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        if self.config.loss.loss_type == "token-wise":
            losses = []
            batch_size = higher_rewards.size(0)
            better_input_ids = concatenated_batch["concatenated_combined_input_ids"][
                :batch_size
            ]
            better_attention_mask = concatenated_batch[
                "concatenated_combined_attention_mask"
            ][:batch_size]
            worse_input_ids = concatenated_batch["concatenated_combined_input_ids"][
                batch_size:
            ]
            worse_attention_mask = concatenated_batch[
                "concatenated_combined_attention_mask"
            ][batch_size:]
            for i in range(batch_size):
                # assert not torch.all(
                #     torch.eq(better_input_ids[i], worse_input_ids[i]),
                # ).item(), "The better and worse answers are the same!"
                # higher_end_index = (
                #     better_attention_mask[i].nonzero()[-1].squeeze().item()
                # )
                # lower_end_index = worse_attention_mask[i].nonzero()[-1].squeeze().item()
                # end_index = max(higher_end_index, lower_end_index)

                # diverge_index = (
                #     (better_input_ids[i] != worse_input_ids[i])
                #     .nonzero()[0]
                #     .squeeze()
                #     .item()
                # )
                # assert 0 <= diverge_index <= end_index, "diverge index is out of range!"

                # # size = (L,)
                # higher_truncated_rewards = higher_rewards[
                #     i, diverge_index : end_index + 1
                # ]
                # lower_truncated_rewards = lower_rewards[
                #     i, diverge_index : end_index + 1
                # ]

                # losses.append(
                #     -F.logsigmoid(
                #         higher_truncated_rewards - lower_truncated_rewards
                #     ).mean(),
                # )
                
                losses.append(
                    -F.logsigmoid(
                        higher_rewards - lower_rewards
                    ).mean(),
                )

                if self.config.regularization > 0.0:
                    losses[-1] = (
                        losses[-1]
                        + self.config.regularization
                        * torch.stack(
                            [lower_rewards, higher_rewards]
                        )
                        .square()
                        .mean()
                    )

            loss = torch.stack(losses).mean()  # size = ()
        elif self.config.loss.loss_type == "sequence-wise":
            loss = -F.logsigmoid(higher_end_reward - lower_end_reward).mean()

            if self.config.loss.regularization > 0.0:
                loss = (
                    loss
                    + self.config.loss.regularization
                    * torch.stack([lower_end_reward, higher_end_reward]).square().mean()
                )
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")

        accuracy = (higher_end_reward > lower_end_reward).float().mean()  # size = ()
        return loss, accuracy

    def get_batch_metrics(
        self, batch: Dict[str, Union[List, torch.LongTensor]], mode: str = None
    ) -> Tuple[torch.FloatTensor, Dict]:
        """Compute the loss and other metrics for the given batch of inputs.

        Arg:
            batch: dictionary of inputs for the batch (what is required will vary depending on the trainer)
            mode: one of 'train', 'eval', 'sample'
        """
        mode = mode or self.config.mode
        (
            concatenated_batch,
            higher_rewards,
            lower_rewards,
            higher_end_reward,
            lower_end_reward,
        ) = self.forward(self.policy, batch)
        loss, accuracy = self.loss(
            concatenated_batch,
            higher_rewards,
            lower_rewards,
            higher_end_reward,
            lower_end_reward,
        )
        higher_end_reward = all_gather_if_needed(higher_end_reward.detach(), self.rank, self.world_size)
        lower_end_reward = all_gather_if_needed(lower_end_reward.detach(), self.rank, self.world_size)
        higher_rewards = all_gather_if_needed(higher_rewards, self.rank, self.world_size)
        lower_rewards = all_gather_if_needed(lower_rewards, self.rank, self.world_size)
        all_devices_losses = all_gather_if_needed(loss.detach(), self.rank, self.world_size)
        all_devices_accuracies = all_gather_if_needed(accuracy, self.rank, self.world_size)
        metrics = {
            f"{mode}/loss": all_devices_losses.float().cpu().numpy().tolist(),
            f"{mode}/accuracy": all_devices_accuracies.float().cpu().numpy().tolist(),
            f"{mode}/higher_end_reward": higher_end_reward.float().cpu().numpy().tolist(),
            f"{mode}/lower_end_reward": lower_end_reward.float().cpu().numpy().tolist(),
            # f"{mode}/higher_rewards": higher_rewards.float().cpu().numpy().tolist(),
            # f"{mode}/lower_rewards": lower_rewards.float().cpu().numpy().tolist(),
        }
        metrics[f"{mode}/all_rewards"] = metrics[f"{mode}/higher_end_reward"] + metrics[f"{mode}/lower_end_reward"]
        del higher_rewards, lower_rewards, higher_end_reward, lower_end_reward, all_devices_accuracies, all_devices_losses

        return loss.mean(), metrics

    def eval(self) -> Dict[str, Dict]:
        """
        Run evaluation on all the examples in the test data and return the metrics from get_batch_metrics.
        This is close-ended evaluation and measures the performance of a single model on a single dataset.
        It does not compare two models to eacch other.

        Returns:
            A dict of form:
            {
                'metadata': the Hydra config
                'results': a dict of batch metrics (averaged across all of the test data)
            }
        """
        rank0_print("Running evaluation")
        self.policy.eval()

        all_eval_metrics = defaultdict(list)

        for eval_batch in (
            tqdm.tqdm(self.eval_batches, desc="Computing eval metrics")
            if self.rank == 0
            else self.eval_batches
        ):
            local_eval_batch = slice_and_move_batch_for_device(
                eval_batch, self.rank, self.world_size, self.rank
            )
            with torch.no_grad():
                _, eval_metrics = self.get_batch_metrics(local_eval_batch, mode="eval")

            for k, v in eval_metrics.items():
                all_eval_metrics[k].extend(v)

        mean_eval_metrics = {}
        for k, v in all_eval_metrics.items():
            if len(v) > 0:
                mean_eval_metrics[k] = sum(v) / len(v)

        results = {
            "metadata": OmegaConf.to_object(self.config),
            "results": formatted_dict(mean_eval_metrics),
        }
        return results

    def sample(self, include_original_prompt=False) -> List[Dict[str, str]]:
        """
        Generate samples from the policy model.

        Args:
            include_original_prompt: whether to include the original prompt among the generated text

        Returns:
            A list of samples, each of which is of the form:
            {
                'prompt': the input
                'chosen': the generation chosen by the human for the given prompt
                'policy': the generation from the policy model
            }
        """
        all_policy_samples, all_prompts, all_chosen, all_original_prompts = (
            [],
            [],
            [],
            [],
        )
        samples = []

        self.policy.eval()
        if self.reference_model is not None:
            self.reference_model.eval()

        for eval_batch in self.eval_batches:
            local_eval_batch = slice_and_move_batch_for_device(
                eval_batch, self.rank, self.world_size, self.rank
            )
            policy_samples = self.get_batch_samples(local_eval_batch)

            chosen_samples = []
            # for DPO-like losses, chosen_text is the field that will contain the text; target_text for all other losses
            # be sure to remove EOS token if present
            for x in (
                eval_batch["target_text"]
                if "target_text" in eval_batch
                else eval_batch["chosen_text"]
            ):
                if self.tokenizer.eos_token in x:
                    x = x[: x.rfind(self.tokenizer.eos_token)]

                chosen_samples.append(x)

            all_prompts.extend(eval_batch["prompt_text"])
            all_original_prompts.extend(eval_batch["original_prompt"])
            all_chosen.extend(chosen_samples)
            all_policy_samples.extend(policy_samples)

            if (
                self.config.n_samples is not None
                and len(all_prompts) > self.config.n_samples
            ):
                break
            else:
                rank0_print(f"Generated {len(all_prompts)} samples ...")

        for i in range(len(all_prompts)):
            if include_original_prompt:
                samples.append(
                    {
                        "prompt": all_prompts[i],
                        "chosen": all_chosen[i],
                        "policy": all_policy_samples[i][
                            len(all_prompts[i]) :
                        ],  # remove the prompt
                        "original_prompt": all_original_prompts[i],
                    }
                )
            else:
                samples.append(
                    {
                        "prompt": all_prompts[i],
                        "chosen": all_chosen[i],
                        "policy": all_policy_samples[i][
                            len(all_prompts[i]) :
                        ],  # remove the prompt
                    }
                )

        return samples

    def train(self):
        """Begin either SFT or HALO training, with periodic evaluation. This is subclassed when implementing PPO."""

        rank0_print(
            f"Using {self.config.optimizer} optimizer with learning rate {self.config.lr}"
        )
        self.optimizer = getattr(torch.optim, self.config.optimizer)(
            self.policy.parameters(), lr=self.config.lr
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(
                1.0, (step + 1) / (self.config.warmup_steps + 1)
            ),
        )

        if self.reference_model is not None:
            self.reference_model.eval()

        last_log = None
        gradients_accumulated = 0
        batch_metrics = defaultdict(list)

        for batch in self.train_iterator:
            # EVALUATION
            if self.example_counter % self.config.eval_every == 0 and (
                self.example_counter > 0 or self.config.do_first_eval
            ):
                rank0_print(
                    f"Running evaluation after {self.example_counter} train examples"
                )
                self.policy.eval()

                all_eval_metrics = defaultdict(list)

                for eval_batch in (
                    tqdm.tqdm(self.eval_batches, desc="Computing eval metrics")
                    if self.rank == 0
                    else self.eval_batches
                ):
                    local_eval_batch = slice_and_move_batch_for_device(
                        eval_batch, self.rank, self.world_size, self.rank
                    )
                    with torch.no_grad():
                        _, eval_metrics = self.get_batch_metrics(
                            local_eval_batch, mode="eval"
                        )

                    for k, v in eval_metrics.items():
                        if isinstance(v, list):
                            all_eval_metrics[k].extend(v)
                        else:
                            all_eval_metrics[k].extend([v])

                    delete_dict(local_eval_batch)

                mean_eval_metrics = {}
                for k, v in all_eval_metrics.items():
                    if len(v) > 0:
                        mean_eval_metrics[k] = np.mean(v)
                rank0_print(
                    f"eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}"
                )

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_eval_metrics, step=self.example_counter)

                if self.example_counter > 0:
                    if self.config.debug:
                        rank0_print("skipping save in debug mode")
                    elif self.config.intermediate_checkpoints:
                        output_dir = os.path.join(
                            self.run_dir, f"step-{self.example_counter}"
                        )
                        rank0_print(f"creating checkpoint to write to {output_dir}...")
                        self.save(output_dir, mean_eval_metrics)

                delete_dict(all_eval_metrics)
                delete_dict(mean_eval_metrics)

            #### TRAINING
            self.policy.train()

            start_time = time.time()

            local_microbatch = slice_and_move_batch_for_device(
                batch, self.rank, self.world_size, self.rank
            )
            loss, metrics = self.get_batch_metrics(local_microbatch)
            (loss / self.config.model.gradient_accumulation_steps).backward()

            for k, v in metrics.items():
                if isinstance(v, list):
                    batch_metrics[k].extend(v)
                else:
                    batch_metrics[k].extend([v])

            gradients_accumulated += 1

            if gradients_accumulated == self.config.model.gradient_accumulation_steps:
                grad_norm = self.clip_gradient()
                batch_metrics["grad_norm"].append(grad_norm)

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                gradients_accumulated = 0

            step_time = time.time() - start_time
            examples_per_second = self.config.model.batch_size / step_time
            batch_metrics["examples_per_second"].append(examples_per_second)

            self.batch_counter += 1
            self.example_counter += self.config.model.batch_size

            delete_dict(local_microbatch)
            delete_dict(metrics)

            if gradients_accumulated == 0 and (
                last_log is None
                or time.time() - last_log > self.config.minimum_log_interval_secs
            ):
                mean_train_metrics = {}
                for k, v in batch_metrics.items():
                    if len(v) > 0:
                        mean_train_metrics[k] = np.mean(v)

                mean_train_metrics["counters/examples"] = self.example_counter
                mean_train_metrics["counters/updates"] = self.batch_counter
                rank0_print(
                    f"train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}"
                )

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()

                delete_dict(batch_metrics)
                delete_dict(mean_train_metrics)
                delete_dict(batch)
                batch_metrics = defaultdict(list)

                # explicitly empty cache if less than 100MB available
                r = torch.cuda.memory_reserved(self.rank)
                a = torch.cuda.memory_allocated(self.rank)

                if (r - a) / 1024 < 100:
                    gc.collect()
                    torch.cuda.empty_cache()
            else:
                rank0_print(
                    f"skipping logging after {self.example_counter} examples to avoid logging too frequently"
                )

    def clip_gradient(self):
        """Clip the gradient norm of the parameters."""
        if self.fsdp:
            a = self.policy.pretrained_model.clip_grad_norm_(self.config.model.max_grad_norm).item()
            b = self.policy.v_head.clip_grad_norm_(self.config.model.max_grad_norm).item()
            return a + b 
        return torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.model.max_grad_norm
        ).item()

    def write_state_dict(
        self,
        step: int,
        state: Dict[str, torch.Tensor],
        metrics: Dict,
        filename: str,
        dir_name: Optional[str] = None,
    ):
        """Write a checkpoint to disk.

        Args:
            step : current training step
            state: current state of training (model or optimizer, if applicable)
            metrics: dictionary of metrics to save
            dir_name: directory in which to save
        """
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, "LATEST")

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f"writing checkpoint to {output_path}...")
        torch.save(
            {
                "step_idx": step,
                "state": state,
                "metrics": metrics if metrics is not None else {},
            },
            output_path,
        )

    def save(
        self,
        output_dir: Optional[str] = None,
        metrics: Optional[Dict] = None,
        save_model_only: bool = True,
    ):
        """
        Save tokenizer, policy model, optimizer, scheduler state to disk, gathering from all processes
        and saving only on the rank 0 process.
        """
        if self.fsdp:
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                self.policy,
                StateDictType.FULL_STATE_DICT,
                state_dict_config=save_policy,
            ):
                policy_state_dict = self.policy.state_dict()

            if self.rank == 0:
                self.write_state_dict(
                    self.example_counter,
                    policy_state_dict,
                    metrics,
                    "reward_model.pt",
                    output_dir,
                )
                self.tokenizer.save_pretrained(
                    self.run_dir
                )  # save tokenizer in HF format

            del policy_state_dict
            dist.barrier()

            if not save_model_only:
                save_policy = FullOptimStateDictConfig(
                    offload_to_cpu=True, rank0_only=True
                )
                with FSDP.state_dict_type(
                    self.policy,
                    StateDictType.FULL_STATE_DICT,
                    optim_state_dict_config=save_policy,
                ):
                    optimizer_state_dict = FSDP.optim_state_dict(
                        self.policy, self.optimizer
                    )

                if self.rank == 0:
                    self.write_state_dict(
                        self.example_counter,
                        optimizer_state_dict,
                        metrics,
                        "optimizer.pt",
                        output_dir,
                    )
                del optimizer_state_dict
                dist.barrier()

                if self.rank == 0:
                    scheduler_state_dict = self.scheduler.state_dict()
                    self.write_state_dict(
                        self.example_counter,
                        scheduler_state_dict,
                        metrics,
                        "scheduler.pt",
                        output_dir,
                    )
                del scheduler_state_dict
                dist.barrier()
        else:
            self.tokenizer.save_pretrained(self.run_dir)  # save tokenizer in HF format
            policy_state_dict = self.policy.state_dict()
            self.write_state_dict(
                self.example_counter,
                policy_state_dict,
                metrics,
                "reward_model.pt",
                output_dir,
            )
            del policy_state_dict

            if not save_model_only:
                optimizer_state_dict = self.optimizer.state_dict()
                self.write_state_dict(
                    self.example_counter,
                    optimizer_state_dict,
                    metrics,
                    "optimizer.pt",
                    output_dir,
                )
                del optimizer_state_dict

                scheduler_state_dict = self.scheduler.state_dict()
                self.write_state_dict(
                    self.example_counter,
                    scheduler_state_dict,
                    metrics,
                    "scheduler.pt",
                    output_dir,
                )
                del scheduler_state_dict


class RLTrainer(object):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        config: DictConfig,
        train_iterator: dataloader.DataLoader,
        eval_iterator: dataloader.DataLoader,
        policy: nn.Module,
        critic=nn.Module,
        reference_model: Optional[nn.Module] = None,
        reward_model: Optional[nn.Module] = None,
        rank: int = 0,
        world_size: int = 1,
        fsdp: bool = False,
    ):
        """A trainer for a language model, supporting either SFT, HALO, or offline PPO training."""
        self.seed = config.seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.rank = rank
        self.device = torch.device("cuda", self.rank)
        self.world_size = world_size
        self.config = config
        self.run_dir = config.local_run_dir
        self.fsdp = fsdp

        self.tokenizer = tokenizer
        self.policy = policy
        self.policy_dtype = getattr(torch, config.model.policy_dtype)
        self.critic = critic
        self.critic_dtype = getattr(torch, config.model.critic_dtype)
        self.reference_model = reference_model
        self.reward_model = reward_model
        self.example_counter = 0
        self.batch_counter = 0

        self.train_iterator = train_iterator
        self.eval_iterator = eval_iterator
        self.eval_batches = list(self.eval_iterator)
        rank0_print(
            f"Loaded {len(self.eval_batches)} eval batches of size {config.model.eval_batch_size}"
        )

        if self.fsdp:
            self.shard()

        self.is_mistral = "mistral" in self.config.model.name_or_path.lower()

    def shard(self):
        """
        Shard the policy model and reference model (if applicable) using FDSP.
        """
        assert (
            self.config.model.block_name is not None
        ), "must specify model.block_name (e.g., GPT2Block or GPTNeoXLayer) for FSDP"
        wrap_class = get_block_class_from_model(
            self.policy,
            self.config.model.block_name,
        )
        model_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={wrap_class},
        )

        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=self.rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False,
        )

        rank0_print("Sharding models...")
        mp_dtype = (
            getattr(torch, self.config.model.fsdp_policy_mp)
            if self.config.model.fsdp_policy_mp is not None
            else None
        )
        mp_policy = MixedPrecision(
            param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype
        )

        # shard the value head according to size
        v_head_shared_fsdp_kwargs = dict(
            auto_wrap_policy=functools.partial(
                size_based_auto_wrap_policy, min_num_params=100
            ),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=self.rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False,
        )

        if self.policy is not None:
            self.policy = FSDP(
                self.policy, **shared_fsdp_kwargs, mixed_precision=mp_policy
            )

        if self.reference_model is not None:
            self.reference_model = FSDP(
                self.reference_model,
                **shared_fsdp_kwargs,
                mixed_precision=mp_policy,
            )
        if self.reward_model is not None:
            self.reward_model.pretrained_model = FSDP(
                self.reward_model.pretrained_model,
                **shared_fsdp_kwargs,
                mixed_precision=mp_policy,
            )
            self.reward_model.v_head = FSDP(
                self.reward_model.v_head, **v_head_shared_fsdp_kwargs
            )
        if self.critic is not None:
            self.critic = FSDP(
                self.critic, **shared_fsdp_kwargs, mixed_precision=mp_policy
            )
            self.critic.v_head = FSDP(self.critic.v_head, **v_head_shared_fsdp_kwargs)

        if self.config.model.activation_checkpointing:
            rank0_print("Attempting to enable activation checkpointing...")
            try:
                # use activation checkpointing, according to:
                # https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/
                # first, verify we have FSDP activation support ready by importing:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    CheckpointImpl,
                    apply_activation_checkpointing,
                    checkpoint_wrapper,
                )
            except Exception as e:
                rank0_print("FSDP activation checkpointing not available:", e)
            else:
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print("Applying activation checkpointing wrapper to policy...")

                if self.policy is not None:
                    apply_activation_checkpointing(
                        self.policy,
                        checkpoint_wrapper_fn=checkpoint_wrapper,
                        check_fn=check_fn,
                    )

                if self.reference_model is not None:
                    apply_activation_checkpointing(
                        self.reference_model,
                        checkpoint_wrapper_fn=checkpoint_wrapper,
                        check_fn=check_fn,
                    )

                if self.reward_model is not None:
                    apply_activation_checkpointing(
                        self.reward_model.pretrained_model,
                        checkpoint_wrapper_fn=checkpoint_wrapper,
                        check_fn=check_fn,
                    )
                if self.critic is not None:
                    apply_activation_checkpointing(
                        self.critic.pretrained_model,
                        checkpoint_wrapper_fn=checkpoint_wrapper,
                        check_fn=check_fn,
                    )

                rank0_print("FSDP activation checkpointing enabled!")

        print("Loaded model on rank", self.rank)
        dist.barrier()

    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        def ctx():
            return (
                FSDP.summon_full_params(self.policy, writeback=False, recurse=False)
                if self.fsdp
                else contextlib.nullcontext()
            )

        with ctx():
            policy_output = self.policy.generate(
                batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.config.model.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                top_p=self.config.top_p,
            )

            policy_output = pad_to_length(
                policy_output, self.config.model.max_length, self.tokenizer.pad_token_id
            )
            policy_output = all_gather_if_needed(
                policy_output, self.rank, self.world_size
            )
            policy_output_decoded = self.tokenizer.batch_decode(
                policy_output, skip_special_tokens=True
            )

        return policy_output_decoded

    def loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the losses, one for each example (sif chosen_only or rejected_only, only n/2 losses).
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively, for reporting.
            Note that rejected responses do not factor into the loss, only the reward calculation.
        """
        raise NotImplementedError

    def get_batch_metrics(
        self, batch: Dict[str, Union[List, torch.LongTensor]], mode: str = None
    ) -> Tuple[torch.FloatTensor, Dict]:
        """Compute the loss and other metrics for the given batch of inputs.

        Arg:
            batch: dictionary of inputs for the batch (what is required will vary depending on the trainer)
            mode: one of 'train', 'eval', 'sample'
        """
        raise NotImplementedError

    def eval(self) -> Dict[str, Dict]:
        """
        Run evaluation on all the examples in the test data and return the metrics from get_batch_metrics.
        This is close-ended evaluation and measures the performance of a single model on a single dataset.
        It does not compare two models to eacch other.

        Returns:
            A dict of form:
            {
                'metadata': the Hydra config
                'results': a dict of batch metrics (averaged across all of the test data)
            }
        """
        rank0_print(f"Running evaluation")
        self.policy.eval()

        if self.reference_model is not None:
            self.reference_model.eval()

        all_eval_metrics = defaultdict(list)

        for eval_batch in (
            tqdm.tqdm(self.eval_batches, desc="Computing eval metrics")
            if self.rank == 0
            else self.eval_batches
        ):
            local_eval_batch = slice_and_move_batch_for_device(
                eval_batch, self.rank, self.world_size, self.rank
            )
            with torch.no_grad():
                _, eval_metrics = self.get_batch_metrics(local_eval_batch, mode="eval")

            for k, v in eval_metrics.items():
                all_eval_metrics[k].extend(v)

        mean_eval_metrics = {}
        for k, v in all_eval_metrics.items():
            if len(v) > 0:
                mean_eval_metrics[k] = sum(v) / len(v)

        results = {
            "metadata": OmegaConf.to_object(self.config),
            "results": formatted_dict(mean_eval_metrics),
        }
        return results

    def sample(self, include_original_prompt=False) -> List[Dict[str, str]]:
        """
        Generate samples from the policy model.

        Args:
            include_original_prompt: whether to include the original prompt among the generated text

        Returns:
            A list of samples, each of which is of the form:
            {
                'prompt': the input
                'chosen': the generation chosen by the human for the given prompt
                'policy': the generation from the policy model
            }
        """
        all_policy_samples, all_prompts, all_chosen, all_original_prompts = (
            [],
            [],
            [],
            [],
        )
        samples = []

        self.policy.eval()
        if self.reference_model is not None:
            self.reference_model.eval()

        for eval_batch in self.eval_batches:
            local_eval_batch = slice_and_move_batch_for_device(
                eval_batch, self.rank, self.world_size, self.rank
            )
            policy_samples = self.get_batch_samples(local_eval_batch)

            chosen_samples = []
            # for DPO-like losses, chosen_text is the field that will contain the text; target_text for all other losses
            # be sure to remove EOS token if present
            for x in (
                eval_batch["target_text"]
                if "target_text" in eval_batch
                else eval_batch["chosen_text"]
            ):
                if self.tokenizer.eos_token in x:
                    x = x[: x.rfind(self.tokenizer.eos_token)]

                chosen_samples.append(x)

            all_prompts.extend(eval_batch["prompt_text"])
            all_original_prompts.extend(eval_batch["original_prompt"])
            all_chosen.extend(chosen_samples)
            all_policy_samples.extend(policy_samples)

            if (
                self.config.n_samples is not None
                and len(all_prompts) > self.config.n_samples
            ):
                break
            else:
                rank0_print(f"Generated {len(all_prompts)} samples ...")

        for i in range(len(all_prompts)):
            if include_original_prompt:
                samples.append(
                    {
                        "prompt": all_prompts[i],
                        "chosen": all_chosen[i],
                        "policy": all_policy_samples[i][
                            len(all_prompts[i]) :
                        ],  # remove the prompt
                        "original_prompt": all_original_prompts[i],
                    }
                )
            else:
                samples.append(
                    {
                        "prompt": all_prompts[i],
                        "chosen": all_chosen[i],
                        "policy": all_policy_samples[i][
                            len(all_prompts[i]) :
                        ],  # remove the prompt
                    }
                )

        return samples

    def train(self):
        """Begin either SFT or HALO training, with periodic evaluation. This is subclassed when implementing PPO."""

        rank0_print(
            f"Using {self.config.optimizer} optimizer with learning rate {self.config.lr}"
        )
        self.optimizer = getattr(torch.optim, self.config.optimizer)(
            self.policy.parameters(), lr=self.config.lr
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(
                1.0, (step + 1) / (self.config.warmup_steps + 1)
            ),
        )

        if self.reference_model is not None:
            self.reference_model.eval()

        last_log = None
        gradients_accumulated = 0
        batch_metrics = defaultdict(list)

        for batch in self.train_iterator:
            # EVALUATION
            if self.example_counter % self.config.eval_every == 0 and (
                self.example_counter > 0 or self.config.do_first_eval
            ):
                rank0_print(
                    f"Running evaluation after {self.example_counter} train examples"
                )
                self.policy.eval()

                all_eval_metrics = defaultdict(list)

                for eval_batch in (
                    tqdm.tqdm(self.eval_batches, desc="Computing eval metrics")
                    if self.rank == 0
                    else self.eval_batches
                ):
                    local_eval_batch = slice_and_move_batch_for_device(
                        eval_batch, self.rank, self.world_size, self.rank
                    )
                    with torch.no_grad():
                        _, eval_metrics = self.get_batch_metrics(
                            local_eval_batch, mode="eval"
                        )

                    for k, v in eval_metrics.items():
                        all_eval_metrics[k].extend(v)

                    delete_dict(local_eval_batch)

                mean_eval_metrics = {}
                for k, v in all_eval_metrics.items():
                    if len(v) > 0:
                        mean_eval_metrics[k] = sum(v) / len(v)
                rank0_print(
                    f"eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}"
                )

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_eval_metrics, step=self.example_counter)

                if self.example_counter > 0:
                    if self.config.debug:
                        rank0_print("skipping save in debug mode")
                    elif self.config.intermediate_checkpoints:
                        output_dir = os.path.join(
                            self.run_dir, f"step-{self.example_counter}"
                        )
                        rank0_print(f"creating checkpoint to write to {output_dir}...")
                        self.save(output_dir, mean_eval_metrics)

                delete_dict(all_eval_metrics)
                delete_dict(mean_eval_metrics)

            #### TRAINING
            self.policy.train()

            start_time = time.time()

            local_microbatch = slice_and_move_batch_for_device(
                batch, self.rank, self.world_size, self.rank
            )
            loss, metrics = self.get_batch_metrics(local_microbatch)
            (loss / self.config.model.gradient_accumulation_steps).backward()

            for k, v in metrics.items():
                batch_metrics[k].extend(v)

            gradients_accumulated += 1

            if gradients_accumulated == self.config.model.gradient_accumulation_steps:
                grad_norm = self.clip_gradient()
                batch_metrics["grad_norm"].append(grad_norm)

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                gradients_accumulated = 0

            step_time = time.time() - start_time
            examples_per_second = self.config.model.batch_size / step_time
            batch_metrics["examples_per_second"].append(examples_per_second)

            self.batch_counter += 1
            self.example_counter += self.config.model.batch_size

            delete_dict(local_microbatch)
            delete_dict(metrics)

            if gradients_accumulated == 0 and (
                last_log is None
                or time.time() - last_log > self.config.minimum_log_interval_secs
            ):
                mean_train_metrics = {}
                for k, v in batch_metrics.items():
                    if len(v) > 0:
                        mean_train_metrics[k] = sum(v) / len(v)

                mean_train_metrics["counters/examples"] = self.example_counter
                mean_train_metrics["counters/updates"] = self.batch_counter
                rank0_print(
                    f"train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}"
                )

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()

                delete_dict(batch_metrics)
                delete_dict(mean_train_metrics)
                delete_dict(batch)
                batch_metrics = defaultdict(list)

                # explicitly empty cache if less than 100MB available
                r = torch.cuda.memory_reserved(self.rank)
                a = torch.cuda.memory_allocated(self.rank)

                if (r - a) / 1024 < 100:
                    gc.collect()
                    torch.cuda.empty_cache()
            else:
                rank0_print(
                    f"skipping logging after {self.example_counter} examples to avoid logging too frequently"
                )

    def clip_gradient(self):
        """Clip the gradient norm of the parameters."""
        if self.fsdp:
            return self.policy.clip_grad_norm_(self.config.model.max_grad_norm).item()
        return torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.model.max_grad_norm
        ).item()

    def write_state_dict(
        self,
        step: int,
        state: Dict[str, torch.Tensor],
        metrics: Dict,
        filename: str,
        dir_name: Optional[str] = None,
    ):
        """Write a checkpoint to disk.

        Args:
            step : current training step
            state: current state of training (model or optimizer, if applicable)
            metrics: dictionary of metrics to save
            dir_name: directory in which to save
        """
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, "LATEST")

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f"writing checkpoint to {output_path}...")
        torch.save(
            {
                "step_idx": step,
                "state": state,
                "metrics": metrics if metrics is not None else {},
            },
            output_path,
        )

    def save(
        self,
        output_dir: Optional[str] = None,
        metrics: Optional[Dict] = None,
        save_model_only: bool = True,
    ):
        """
        Save tokenizer, policy model, optimizer, scheduler state to disk, gathering from all processes
        and saving only on the rank 0 process.
        """
        if self.fsdp:
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                self.policy,
                StateDictType.FULL_STATE_DICT,
                state_dict_config=save_policy,
            ):
                policy_state_dict = self.policy.state_dict()

            if self.rank == 0:
                self.write_state_dict(
                    self.example_counter,
                    policy_state_dict,
                    metrics,
                    "policy.pt",
                    output_dir,
                )
                self.tokenizer.save_pretrained(
                    self.run_dir
                )  # save tokenizer in HF format

            del policy_state_dict
            dist.barrier()

            if not save_model_only:
                save_policy = FullOptimStateDictConfig(
                    offload_to_cpu=True, rank0_only=True
                )
                with FSDP.state_dict_type(
                    self.policy,
                    StateDictType.FULL_STATE_DICT,
                    optim_state_dict_config=save_policy,
                ):
                    optimizer_state_dict = FSDP.optim_state_dict(
                        self.policy, self.optimizer
                    )

                if self.rank == 0:
                    self.write_state_dict(
                        self.example_counter,
                        optimizer_state_dict,
                        metrics,
                        "optimizer.pt",
                        output_dir,
                    )
                del optimizer_state_dict
                dist.barrier()

                if self.rank == 0:
                    scheduler_state_dict = self.scheduler.state_dict()
                    self.write_state_dict(
                        self.example_counter,
                        scheduler_state_dict,
                        metrics,
                        "scheduler.pt",
                        output_dir,
                    )
                del scheduler_state_dict
                dist.barrier()
        else:
            self.tokenizer.save_pretrained(self.run_dir)  # save tokenizer in HF format
            policy_state_dict = self.policy.state_dict()
            self.write_state_dict(
                self.example_counter,
                policy_state_dict,
                metrics,
                "policy.pt",
                output_dir,
            )
            del policy_state_dict

            if not save_model_only:
                optimizer_state_dict = self.optimizer.state_dict()
                self.write_state_dict(
                    self.example_counter,
                    optimizer_state_dict,
                    metrics,
                    "optimizer.pt",
                    output_dir,
                )
                del optimizer_state_dict

                scheduler_state_dict = self.scheduler.state_dict()
                self.write_state_dict(
                    self.example_counter,
                    scheduler_state_dict,
                    metrics,
                    "scheduler.pt",
                    output_dir,
                )
                del scheduler_state_dict

# Copyright (c) 2023 Contextual AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Main script for training.

Sample use is:

python train.py loss=ppo model=llama30b datasets=[shp,hh,oasst] exp_name=archangel_sft+ppo_llama30b mode=train \
     ++cache_dir=/data/models/archangel ++model.load_from=archangel_sft_llama30b/LATEST/policy.pt

where
- loss should have a file under config/loss that specifies the trainer in trainers.py and dataloader in dataloader.py
- model should have a file under config/model
- datasets is a list of datasets, each of which has a get_{name} function in dataloader.py
- exp_name is the experiment name (on WANDB); model will be saved to the cache_dir/exp_name
- model.load_from should be used for aligning a model that has already been finetuned

Remember to allocate enough RAM before running this (you need aroundd 800 GB for Llama-13B).
"""

import gc
import json
import os
import resource
import socket
from typing import Optional, Set

import hydra
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, set_seed

import dataloader
import rl_trainers
from ppo_models import AutoModelForCausalLMWithValueHead
from utils import delete_dict, disable_dropout, get_open_port, init_distributed

torch.backends.cuda.matmul.allow_tf32 = True


def worker_main(
    rank: int,
    world_size: int,
    config: DictConfig,
    tokenizer: AutoTokenizer,
    train_iterator: dataloader.DataLoader,
    eval_iterator: dataloader.DataLoader,
    policy: nn.Module,
    reference_model: Optional[nn.Module] = None,
):
    """Main function for each worker process (may be only 1 for BasicTrainer)."""
    if config.use_fsdp:
        init_distributed(rank, world_size, port=config.fsdp_port)

    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None

    if rank == 0 and config.wandb.enabled:
        os.environ["WANDB_CACHE_DIR"] = config.cache_dir
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=config.cache_dir,
            name=config.exp_name,
        )

    TrainerClass = getattr(rl_trainers, config.loss.trainer)
    print(f"Creating trainer on process {rank} with world size {world_size}")

    trainer = TrainerClass(
        tokenizer,
        config,
        train_iterator,
        eval_iterator,
        policy,
        reference_model=reference_model,
        rank=rank,
        world_size=world_size,
        fsdp=config.use_fsdp,
    )

    trainer.train()
    trainer.save()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""
    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    os.makedirs(config.local_run_dir, exist_ok=True)
    print("Making experiment directory", config.local_run_dir)

    set_seed(config.seed)

    if config.eval_every % config.model.batch_size != 0:
        print("WARNING: eval_every must be divisible by batch_size")
        print(
            "Setting eval_every to",
            config.eval_every - config.eval_every % config.model.batch_size,
        )
        config.eval_every = (
            config.eval_every - config.eval_every % config.model.batch_size
        )

    if config.use_fsdp and config.fsdp_port is None:
        free_port = get_open_port()
        print("no FSDP port specified; using open port for FSDP:", free_port)
        config.fsdp_port = free_port

    if config.saved_policy is None:
        config.saved_policy = f"{config.local_run_dir}/LATEST/policy.pt"

    print(OmegaConf.to_yaml(config))

    config_path = os.path.join(config.local_run_dir, "config.yaml")
    with open(config_path, "w") as f:
        OmegaConf.save(config, f)

    print("=" * 80)
    print(f"Writing to {socket.gethostname()}:{config.local_run_dir}")
    print("=" * 80)

    reward_model_kwargs = {
        "torch_dtype": getattr(torch, config.model.policy_dtype),
    }

    if not config.use_fsdp:
        reward_model_kwargs["device_map"] = "auto"

    print("building reward model")
    reward_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model.name_or_path,
        low_cpu_mem_usage=True,
        use_flash_attention_2=config.model.use_flash_attention,
        **reward_model_kwargs,
    )
    disable_dropout(reward_model)
    if not config.use_fsdp:
        reward_model = reward_model.to("cuda")

    if config.model.load_reward_model_from is not None:
        state_dict = torch.load(
            os.path.join(config.cache_dir, config.model.load_reward_model_from),
            map_location="cpu",
        )
        step, metrics = state_dict["step_idx"], state_dict["metrics"]
        print(
            f"loading pre-trained weights of RM at step {step} from {config.model.load_reward_model_from} with metrics {json.dumps(metrics, indent=2)}"
        )

        reward_model.load_state_dict(state_dict["state"])

        delete_dict(state_dict)
        gc.collect()
        torch.cuda.empty_cache()

        print("loaded RM/CRITIC pre-trained weights")

    tokenizer_name_or_path = (
        config.model.tokenizer_name_or_path or config.model.name_or_path
    )
    print(f"Loading tokenizer {tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    data_loader_class = getattr(dataloader, config.loss.dataloader)
    data_iterator_kwargs = dict(
        max_length=config.model.max_length,
        max_prompt_length=config.model.max_prompt_length,
        human_prefix=config.human_prefix,
        human_suffix=config.human_suffix,
        assistant_prefix=config.assistant_prefix,
        assistant_suffix=config.assistant_suffix,
        seed=config.seed,
        frac_unique_desirable=config.frac_unique_desirable,
        frac_unique_undesirable=config.frac_unique_undesirable,
        # control tokens taken from Korbak et al.'s (2023) "Pretraining Models with Human Feedback"
        # SFTDataLoader will use them for sampling; ConditionalSFTDataLoader for training
        chosen_control_token=(
            config.loss.chosen_control_token if config.loss.name == "csft" else None
        ),
        rejected_control_token=(
            config.loss.rejected_control_token if config.loss.name == "csft" else None
        ),
    )
    train_iterator = data_loader_class(
        config.datasets,
        tokenizer,
        split="train",
        batch_size=config.model.batch_size,
        n_epochs=config.n_epochs,
        n_examples=config.n_examples,
        **data_iterator_kwargs,
    )
    eval_iterator = data_loader_class(
        config.datasets,
        tokenizer,
        split="test",
        batch_size=config.model.eval_batch_size,
        n_examples=config.n_eval_examples,
        n_epochs=(1 if config.n_eval_examples is None else None),
        **data_iterator_kwargs,
    )

    if config.use_fsdp:
        world_size = torch.cuda.device_count()
        print("starting", world_size, "processes for FSDP training")
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f"setting RLIMIT_NOFILE soft limit to {hard} from {soft}")
        mp.spawn(
            worker_main,
            nprocs=world_size,
            args=(
                world_size,
                config,
                tokenizer,
                train_iterator,
                eval_iterator,
                reward_model,
                None,
            ),
            join=True,
        )
    else:
        print("starting single-process worker")
        worker_main(
            0,
            1,
            config,
            tokenizer,
            train_iterator,
            eval_iterator,
            reward_model,
            None,
        )


if __name__ == "__main__":
    main()

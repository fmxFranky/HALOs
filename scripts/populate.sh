#!/bin/bash

# Sample use is './scripts/populate.sh dpo llama7b'. 
# If the model isn't specified, then it will run on all models. If the loss isn't specified, it will run all losses on all models.

cache_dir="/data/models/archangel"
nargs="$#"
losses=("sft" "sft+csft" "sft+dpo" "sft+kto" "sft+ppo" "sft+slic" "csft" "dpo" "kto" "ppo" "slic")
models=("pythia1-4b" "pythia2-8b" "pythia6-9b" "pythia12-0b" "llama7b" "llama13b" "llama30b")

if [ $nargs == 1 ]; then
    losses=("$1")
elif [ $nargs == 2 ]; then
    losses=("$1")
    models=("$2")
fi

for loss in "${losses[@]}"; do
    for model in "${models[@]}"; do
        exp_name="archangel_${loss}_${model}"
        echo "$exp_name"

        # llama30 has to use smaller batch sizes + gradient accumulation to get same effective batch size
        if [ "$model" == "llama30b" ]; then
            if [[ $loss == "sft+"* ]]; then
                sft_model="archangel_sft_${model}/LATEST/policy.pt"
                alignment_loss="${loss:4}"
                python train.py loss="$alignment_loss" model="$model" datasets=[shp,hh,oasst] exp_name="$exp_name" mode=train ++cache_dir="$cache_dir" ++model.load_from="$sft_model" ++model.batch_size=16 ++model.gradient_accumulation_steps=2
            elif [[ $loss == "sft" ]]; then
                python train.py loss="$loss" model="$model" datasets=[shp,hh,oasst] exp_name="$exp_name" mode=train ++cache_dir="$cache_dir"
            else
                python train.py loss="$loss" model="$model" datasets=[shp,hh,oasst] exp_name="$exp_name" mode=train ++cache_dir="$cache_dir" ++model.batch_size=16 ++model.gradient_accumulation_steps=2
            fi
        else
            if [[ $loss == "sft+"* ]]; then
                sft_model="archangel_sft_${model}/LATEST/policy.pt"
                alignment_loss="${loss:4}"
                python train.py loss="$alignment_loss" model="$model" datasets=[shp,hh,oasst] exp_name="$exp_name" mode=train ++cache_dir="$cache_dir" ++model.load_from="$sft_model"
            else
                python train.py loss="$loss" model="$model" datasets=[shp,hh,oasst] exp_name="$exp_name" mode=train ++cache_dir="$cache_dir"
            fi
        fi
    done
done
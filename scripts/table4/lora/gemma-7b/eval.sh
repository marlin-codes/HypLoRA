#!/bin/bash
python commonsense_evaluate_gemma.py \
    --model 'gemma-7b' --adapter LoRA \
    --dataset 'boolq' \
    --base_model 'google/gemma-7b' \
    --lora_weights './trained_models/gemma-7b-lora-r32-commonsense' \
    --rank 32 --lora_type 'std' \
    --lora_alpha 64 --batch_size 1
python commonsense_evaluate_gemma.py \
    --model 'gemma-7b' --adapter LoRA \
    --dataset 'piqa' \
    --base_model 'google/gemma-7b' \
    --lora_weights './trained_models/gemma-7b-lora-r32-commonsense' \
    --rank 32 --lora_type 'std' \
    --lora_alpha 64 --batch_size 1
python commonsense_evaluate_gemma.py \
    --model 'gemma-7b' --adapter LoRA \
    --dataset 'social_i_qa' \
    --base_model 'google/gemma-7b' \
    --lora_weights './trained_models/gemma-7b-lora-r32-commonsense' \
    --rank 32 --lora_type 'std' \
    --lora_alpha 64 --batch_size 1
python commonsense_evaluate_gemma.py \
    --model 'gemma-7b' --adapter LoRA \
    --dataset 'hellaswag' \
    --base_model 'google/gemma-7b' \
    --lora_weights './trained_models/gemma-7b-lora-r32-commonsense' \
    --rank 32 --lora_type 'std' \
    --lora_alpha 64 --batch_size 1
python commonsense_evaluate_gemma.py \
    --model 'gemma-7b' --adapter LoRA \
    --dataset 'winogrande' \
    --base_model 'google/gemma-7b' \
    --lora_weights './trained_models/gemma-7b-lora-r32-commonsense' \
    --rank 32 --lora_type 'std' \
    --lora_alpha 64 --batch_size 1
python commonsense_evaluate_gemma.py \
    --model 'gemma-7b' --adapter LoRA \
    --dataset 'ARC-Easy' \
    --base_model 'google/gemma-7b' \
    --lora_weights './trained_models/gemma-7b-lora-r32-commonsense' \
    --rank 32 --lora_type 'std' \
    --lora_alpha 64 --batch_size 1
python commonsense_evaluate_gemma.py \
    --model 'gemma-7b' --adapter LoRA \
    --dataset 'ARC-Challenge' \
    --base_model 'google/gemma-7b' \
    --lora_weights './trained_models/gemma-7b-lora-r32-commonsense' \
    --rank 32 --lora_type 'std' \
    --lora_alpha 64 --batch_size 1
python commonsense_evaluate_gemma.py \
    --model 'gemma-7b' --adapter LoRA \
    --dataset 'openbookqa' \
    --base_model 'google/gemma-7b' \
    --lora_weights './trained_models/gemma-7b-lora-r32-commonsense' \
    --rank 32 --lora_type 'std' \
    --lora_alpha 64 --batch_size 1

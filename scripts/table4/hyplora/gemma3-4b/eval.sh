#!/bin/bash
python commonsense_evaluate_gemma.py \
    --model 'gemma-3-4b-it' --adapter LoRA \
    --dataset 'boolq' \
    --base_model 'google/gemma-3-4b-it' \
    --lora_weights './trained_models/gemma3-4b-hyplora-k1.0-r32-commonsense' \
    --rank 32 --lora_type 'hyplora-1.0' \
    --lora_alpha 32 --batch_size 1
python commonsense_evaluate_gemma.py \
    --model 'gemma-3-4b-it' --adapter LoRA \
    --dataset 'piqa' \
    --base_model 'google/gemma-3-4b-it' \
    --lora_weights './trained_models/gemma3-4b-hyplora-k1.0-r32-commonsense' \
    --rank 32 --lora_type 'hyplora-1.0' \
    --lora_alpha 32 --batch_size 1
python commonsense_evaluate_gemma.py \
    --model 'gemma-3-4b-it' --adapter LoRA \
    --dataset 'social_i_qa' \
    --base_model 'google/gemma-3-4b-it' \
    --lora_weights './trained_models/gemma3-4b-hyplora-k1.0-r32-commonsense' \
    --rank 32 --lora_type 'hyplora-1.0' \
    --lora_alpha 32 --batch_size 1
python commonsense_evaluate_gemma.py \
    --model 'gemma-3-4b-it' --adapter LoRA \
    --dataset 'hellaswag' \
    --base_model 'google/gemma-3-4b-it' \
    --lora_weights './trained_models/gemma3-4b-hyplora-k1.0-r32-commonsense' \
    --rank 32 --lora_type 'hyplora-1.0' \
    --lora_alpha 32 --batch_size 1
python commonsense_evaluate_gemma.py \
    --model 'gemma-3-4b-it' --adapter LoRA \
    --dataset 'winogrande' \
    --base_model 'google/gemma-3-4b-it' \
    --lora_weights './trained_models/gemma3-4b-hyplora-k1.0-r32-commonsense' \
    --rank 32 --lora_type 'hyplora-1.0' \
    --lora_alpha 32 --batch_size 1
python commonsense_evaluate_gemma.py \
    --model 'gemma-3-4b-it' --adapter LoRA \
    --dataset 'ARC-Easy' \
    --base_model 'google/gemma-3-4b-it' \
    --lora_weights './trained_models/gemma3-4b-hyplora-k1.0-r32-commonsense' \
    --rank 32 --lora_type 'hyplora-1.0' \
    --lora_alpha 32 --batch_size 1
python commonsense_evaluate_gemma.py \
    --model 'gemma-3-4b-it' --adapter LoRA \
    --dataset 'ARC-Challenge' \
    --base_model 'google/gemma-3-4b-it' \
    --lora_weights './trained_models/gemma3-4b-hyplora-k1.0-r32-commonsense' \
    --rank 32 --lora_type 'hyplora-1.0' \
    --lora_alpha 32 --batch_size 1
python commonsense_evaluate_gemma.py \
    --model 'gemma-3-4b-it' --adapter LoRA \
    --dataset 'openbookqa' \
    --base_model 'google/gemma-3-4b-it' \
    --lora_weights './trained_models/gemma3-4b-hyplora-k1.0-r32-commonsense' \
    --rank 32 --lora_type 'hyplora-1.0' \
    --lora_alpha 32 --batch_size 1

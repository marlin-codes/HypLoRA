#!/bin/bash
python evaluate.py \
    --dataset 'AQuA' --model 'gemma-7b' \
    --base_model 'google/gemma-7b' \
    --lora_weights './trained_models/gemma-7b-lora-r32-math' \
    --lora_type 'std' --lora_alpha 64 \
    --rank 32 --adapter 'LoRA' --run 0
python evaluate.py \
    --dataset 'mawps' --model 'gemma-7b' \
    --base_model 'google/gemma-7b' \
    --lora_weights './trained_models/gemma-7b-lora-r32-math' \
    --lora_type 'std' --lora_alpha 64 \
    --rank 32 --adapter 'LoRA' --run 0
python evaluate.py \
    --dataset 'SVAMP' --model 'gemma-7b' \
    --base_model 'google/gemma-7b' \
    --lora_weights './trained_models/gemma-7b-lora-r32-math' \
    --lora_type 'std' --lora_alpha 64 \
    --rank 32 --adapter 'LoRA' --run 0
python evaluate.py \
    --dataset 'gsm8k' --model 'gemma-7b' \
    --base_model 'google/gemma-7b' \
    --lora_weights './trained_models/gemma-7b-lora-r32-math' \
    --lora_type 'std' --lora_alpha 64 \
    --rank 32 --adapter 'LoRA' --run 0

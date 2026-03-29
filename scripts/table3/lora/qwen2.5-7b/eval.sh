#!/bin/bash
python evaluate.py \
    --dataset 'AQuA' --model 'Qwen2.5-7B-Instruct-1M' \
    --base_model 'Qwen/Qwen2.5-7B-Instruct-1M' \
    --lora_weights './trained_models/qwen2.5-7b-lora-r32-math' \
    --lora_type 'std' --lora_alpha 64 \
    --rank 32 --adapter 'LoRA' --run 0
python evaluate.py \
    --dataset 'mawps' --model 'Qwen2.5-7B-Instruct-1M' \
    --base_model 'Qwen/Qwen2.5-7B-Instruct-1M' \
    --lora_weights './trained_models/qwen2.5-7b-lora-r32-math' \
    --lora_type 'std' --lora_alpha 64 \
    --rank 32 --adapter 'LoRA' --run 0
python evaluate.py \
    --dataset 'SVAMP' --model 'Qwen2.5-7B-Instruct-1M' \
    --base_model 'Qwen/Qwen2.5-7B-Instruct-1M' \
    --lora_weights './trained_models/qwen2.5-7b-lora-r32-math' \
    --lora_type 'std' --lora_alpha 64 \
    --rank 32 --adapter 'LoRA' --run 0
python evaluate.py \
    --dataset 'gsm8k' --model 'Qwen2.5-7B-Instruct-1M' \
    --base_model 'Qwen/Qwen2.5-7B-Instruct-1M' \
    --lora_weights './trained_models/qwen2.5-7b-lora-r32-math' \
    --lora_type 'std' --lora_alpha 64 \
    --rank 32 --adapter 'LoRA' --run 0

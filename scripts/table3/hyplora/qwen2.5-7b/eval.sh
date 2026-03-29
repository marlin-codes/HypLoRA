#!/bin/bash
python evaluate.py \
    --dataset 'AQuA' --model 'Qwen2.5-7B-Instruct-1M' \
    --base_model 'Qwen/Qwen2.5-7B-Instruct-1M' \
    --lora_weights './trained_models/qwen2.5-7b-hyplora-k1.0-r32-math' \
    --lora_type 'hyplora-1.0' --lora_alpha 128 \
    --rank 32 --adapter 'LoRA' --run 0
python evaluate.py \
    --dataset 'mawps' --model 'Qwen2.5-7B-Instruct-1M' \
    --base_model 'Qwen/Qwen2.5-7B-Instruct-1M' \
    --lora_weights './trained_models/qwen2.5-7b-hyplora-k1.0-r32-math' \
    --lora_type 'hyplora-1.0' --lora_alpha 128 \
    --rank 32 --adapter 'LoRA' --run 0
python evaluate.py \
    --dataset 'SVAMP' --model 'Qwen2.5-7B-Instruct-1M' \
    --base_model 'Qwen/Qwen2.5-7B-Instruct-1M' \
    --lora_weights './trained_models/qwen2.5-7b-hyplora-k1.0-r32-math' \
    --lora_type 'hyplora-1.0' --lora_alpha 128 \
    --rank 32 --adapter 'LoRA' --run 0
python evaluate.py \
    --dataset 'gsm8k' --model 'Qwen2.5-7B-Instruct-1M' \
    --base_model 'Qwen/Qwen2.5-7B-Instruct-1M' \
    --lora_weights './trained_models/qwen2.5-7b-hyplora-k1.0-r32-math' \
    --lora_type 'hyplora-1.0' --lora_alpha 128 \
    --rank 32 --adapter 'LoRA' --run 0

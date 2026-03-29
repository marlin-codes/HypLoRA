#!/bin/bash
python evaluate.py \
    --dataset 'AQuA' --model 'Qwen2.5-7B-Instruct-1M' \
    --base_model 'Qwen/Qwen2.5-7B-Instruct-1M' \
    --lora_weights './trained_models/qwen2.5-7b-hyplora-simplified-c0.5-r32-math' \
    --lora_type 'hyplora_simplified-0.5' --lora_alpha 256 \
    --rank 32 --adapter 'LoRA' --run 0
python evaluate.py \
    --dataset 'mawps' --model 'Qwen2.5-7B-Instruct-1M' \
    --base_model 'Qwen/Qwen2.5-7B-Instruct-1M' \
    --lora_weights './trained_models/qwen2.5-7b-hyplora-simplified-c0.5-r32-math' \
    --lora_type 'hyplora_simplified-0.5' --lora_alpha 256 \
    --rank 32 --adapter 'LoRA' --run 0
python evaluate.py \
    --dataset 'SVAMP' --model 'Qwen2.5-7B-Instruct-1M' \
    --base_model 'Qwen/Qwen2.5-7B-Instruct-1M' \
    --lora_weights './trained_models/qwen2.5-7b-hyplora-simplified-c0.5-r32-math' \
    --lora_type 'hyplora_simplified-0.5' --lora_alpha 256 \
    --rank 32 --adapter 'LoRA' --run 0
python evaluate.py \
    --dataset 'gsm8k' --model 'Qwen2.5-7B-Instruct-1M' \
    --base_model 'Qwen/Qwen2.5-7B-Instruct-1M' \
    --lora_weights './trained_models/qwen2.5-7b-hyplora-simplified-c0.5-r32-math' \
    --lora_type 'hyplora_simplified-0.5' --lora_alpha 256 \
    --rank 32 --adapter 'LoRA' --run 0

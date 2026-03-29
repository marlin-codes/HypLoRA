#!/bin/bash
python evaluate.py \
    --dataset 'AQuA' --model 'Meta-Llama-3-8B-Instruct' \
    --base_model 'meta-llama/Meta-Llama-3-8B-Instruct' \
    --lora_weights './trained_models/llama3-8b-hyplora-simplified-c0.5-r32-math' \
    --lora_type 'hyplora_simplified-0.5' --lora_alpha 256 \
    --rank 32 --adapter 'LoRA' --run 0
python evaluate.py \
    --dataset 'mawps' --model 'Meta-Llama-3-8B-Instruct' \
    --base_model 'meta-llama/Meta-Llama-3-8B-Instruct' \
    --lora_weights './trained_models/llama3-8b-hyplora-simplified-c0.5-r32-math' \
    --lora_type 'hyplora_simplified-0.5' --lora_alpha 256 \
    --rank 32 --adapter 'LoRA' --run 0
python evaluate.py \
    --dataset 'SVAMP' --model 'Meta-Llama-3-8B-Instruct' \
    --base_model 'meta-llama/Meta-Llama-3-8B-Instruct' \
    --lora_weights './trained_models/llama3-8b-hyplora-simplified-c0.5-r32-math' \
    --lora_type 'hyplora_simplified-0.5' --lora_alpha 256 \
    --rank 32 --adapter 'LoRA' --run 0
python evaluate.py \
    --dataset 'gsm8k' --model 'Meta-Llama-3-8B-Instruct' \
    --base_model 'meta-llama/Meta-Llama-3-8B-Instruct' \
    --lora_weights './trained_models/llama3-8b-hyplora-simplified-c0.5-r32-math' \
    --lora_type 'hyplora_simplified-0.5' --lora_alpha 256 \
    --rank 32 --adapter 'LoRA' --run 0

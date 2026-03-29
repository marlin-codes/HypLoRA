#!/bin/bash
python evaluate.py \
    --dataset 'AQuA' --model 'Meta-Llama-3-8B-Instruct' \
    --base_model 'meta-llama/Meta-Llama-3-8B-Instruct' \
    --lora_weights './trained_models/llama3-8b-hyplora-k1.0-r32-math' \
    --lora_type 'hyplora-1.0' --lora_alpha 128 \
    --rank 32 --adapter 'LoRA' --run 0
python evaluate.py \
    --dataset 'mawps' --model 'Meta-Llama-3-8B-Instruct' \
    --base_model 'meta-llama/Meta-Llama-3-8B-Instruct' \
    --lora_weights './trained_models/llama3-8b-hyplora-k1.0-r32-math' \
    --lora_type 'hyplora-1.0' --lora_alpha 128 \
    --rank 32 --adapter 'LoRA' --run 0
python evaluate.py \
    --dataset 'SVAMP' --model 'Meta-Llama-3-8B-Instruct' \
    --base_model 'meta-llama/Meta-Llama-3-8B-Instruct' \
    --lora_weights './trained_models/llama3-8b-hyplora-k1.0-r32-math' \
    --lora_type 'hyplora-1.0' --lora_alpha 128 \
    --rank 32 --adapter 'LoRA' --run 0
python evaluate.py \
    --dataset 'gsm8k' --model 'Meta-Llama-3-8B-Instruct' \
    --base_model 'meta-llama/Meta-Llama-3-8B-Instruct' \
    --lora_weights './trained_models/llama3-8b-hyplora-k1.0-r32-math' \
    --lora_type 'hyplora-1.0' --lora_alpha 128 \
    --rank 32 --adapter 'LoRA' --run 0

#!/bin/bash
python commonsense_evaluate_qwen.py \
    --model 'Qwen2.5-7B-Instruct-1M' --adapter LoRA \
    --dataset 'boolq' \
    --base_model 'Qwen/Qwen2.5-7B-Instruct-1M' \
    --lora_weights './trained_models/qwen2.5-7b-hyplora-k1.0-r32-commonsense' \
    --rank 32 --lora_type 'hyplora-1.0' \
    --lora_alpha 256 --batch_size 1
python commonsense_evaluate_qwen.py \
    --model 'Qwen2.5-7B-Instruct-1M' --adapter LoRA \
    --dataset 'piqa' \
    --base_model 'Qwen/Qwen2.5-7B-Instruct-1M' \
    --lora_weights './trained_models/qwen2.5-7b-hyplora-k1.0-r32-commonsense' \
    --rank 32 --lora_type 'hyplora-1.0' \
    --lora_alpha 256 --batch_size 1
python commonsense_evaluate_qwen.py \
    --model 'Qwen2.5-7B-Instruct-1M' --adapter LoRA \
    --dataset 'social_i_qa' \
    --base_model 'Qwen/Qwen2.5-7B-Instruct-1M' \
    --lora_weights './trained_models/qwen2.5-7b-hyplora-k1.0-r32-commonsense' \
    --rank 32 --lora_type 'hyplora-1.0' \
    --lora_alpha 256 --batch_size 1
python commonsense_evaluate_qwen.py \
    --model 'Qwen2.5-7B-Instruct-1M' --adapter LoRA \
    --dataset 'hellaswag' \
    --base_model 'Qwen/Qwen2.5-7B-Instruct-1M' \
    --lora_weights './trained_models/qwen2.5-7b-hyplora-k1.0-r32-commonsense' \
    --rank 32 --lora_type 'hyplora-1.0' \
    --lora_alpha 256 --batch_size 1
python commonsense_evaluate_qwen.py \
    --model 'Qwen2.5-7B-Instruct-1M' --adapter LoRA \
    --dataset 'winogrande' \
    --base_model 'Qwen/Qwen2.5-7B-Instruct-1M' \
    --lora_weights './trained_models/qwen2.5-7b-hyplora-k1.0-r32-commonsense' \
    --rank 32 --lora_type 'hyplora-1.0' \
    --lora_alpha 256 --batch_size 1
python commonsense_evaluate_qwen.py \
    --model 'Qwen2.5-7B-Instruct-1M' --adapter LoRA \
    --dataset 'ARC-Easy' \
    --base_model 'Qwen/Qwen2.5-7B-Instruct-1M' \
    --lora_weights './trained_models/qwen2.5-7b-hyplora-k1.0-r32-commonsense' \
    --rank 32 --lora_type 'hyplora-1.0' \
    --lora_alpha 256 --batch_size 1
python commonsense_evaluate_qwen.py \
    --model 'Qwen2.5-7B-Instruct-1M' --adapter LoRA \
    --dataset 'ARC-Challenge' \
    --base_model 'Qwen/Qwen2.5-7B-Instruct-1M' \
    --lora_weights './trained_models/qwen2.5-7b-hyplora-k1.0-r32-commonsense' \
    --rank 32 --lora_type 'hyplora-1.0' \
    --lora_alpha 256 --batch_size 1
python commonsense_evaluate_qwen.py \
    --model 'Qwen2.5-7B-Instruct-1M' --adapter LoRA \
    --dataset 'openbookqa' \
    --base_model 'Qwen/Qwen2.5-7B-Instruct-1M' \
    --lora_weights './trained_models/qwen2.5-7b-hyplora-k1.0-r32-commonsense' \
    --rank 32 --lora_type 'hyplora-1.0' \
    --lora_alpha 256 --batch_size 1

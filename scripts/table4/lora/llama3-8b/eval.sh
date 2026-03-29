#!/bin/bash
python commonsense_evaluate.py \
    --model 'Meta-Llama-3-8B-Instruct' --adapter LoRA \
    --dataset 'boolq' \
    --base_model 'meta-llama/Meta-Llama-3-8B-Instruct' \
    --lora_weights './trained_models/llama3-8b-lora-r32-commonsense' \
    --rank 32 --lora_type 'std' \
    --lora_alpha 64 --batch_size 1
python commonsense_evaluate.py \
    --model 'Meta-Llama-3-8B-Instruct' --adapter LoRA \
    --dataset 'piqa' \
    --base_model 'meta-llama/Meta-Llama-3-8B-Instruct' \
    --lora_weights './trained_models/llama3-8b-lora-r32-commonsense' \
    --rank 32 --lora_type 'std' \
    --lora_alpha 64 --batch_size 1
python commonsense_evaluate.py \
    --model 'Meta-Llama-3-8B-Instruct' --adapter LoRA \
    --dataset 'social_i_qa' \
    --base_model 'meta-llama/Meta-Llama-3-8B-Instruct' \
    --lora_weights './trained_models/llama3-8b-lora-r32-commonsense' \
    --rank 32 --lora_type 'std' \
    --lora_alpha 64 --batch_size 1
python commonsense_evaluate.py \
    --model 'Meta-Llama-3-8B-Instruct' --adapter LoRA \
    --dataset 'hellaswag' \
    --base_model 'meta-llama/Meta-Llama-3-8B-Instruct' \
    --lora_weights './trained_models/llama3-8b-lora-r32-commonsense' \
    --rank 32 --lora_type 'std' \
    --lora_alpha 64 --batch_size 1
python commonsense_evaluate.py \
    --model 'Meta-Llama-3-8B-Instruct' --adapter LoRA \
    --dataset 'winogrande' \
    --base_model 'meta-llama/Meta-Llama-3-8B-Instruct' \
    --lora_weights './trained_models/llama3-8b-lora-r32-commonsense' \
    --rank 32 --lora_type 'std' \
    --lora_alpha 64 --batch_size 1
python commonsense_evaluate.py \
    --model 'Meta-Llama-3-8B-Instruct' --adapter LoRA \
    --dataset 'ARC-Easy' \
    --base_model 'meta-llama/Meta-Llama-3-8B-Instruct' \
    --lora_weights './trained_models/llama3-8b-lora-r32-commonsense' \
    --rank 32 --lora_type 'std' \
    --lora_alpha 64 --batch_size 1
python commonsense_evaluate.py \
    --model 'Meta-Llama-3-8B-Instruct' --adapter LoRA \
    --dataset 'ARC-Challenge' \
    --base_model 'meta-llama/Meta-Llama-3-8B-Instruct' \
    --lora_weights './trained_models/llama3-8b-lora-r32-commonsense' \
    --rank 32 --lora_type 'std' \
    --lora_alpha 64 --batch_size 1
python commonsense_evaluate.py \
    --model 'Meta-Llama-3-8B-Instruct' --adapter LoRA \
    --dataset 'openbookqa' \
    --base_model 'meta-llama/Meta-Llama-3-8B-Instruct' \
    --lora_weights './trained_models/llama3-8b-lora-r32-commonsense' \
    --rank 32 --lora_type 'std' \
    --lora_alpha 64 --batch_size 1

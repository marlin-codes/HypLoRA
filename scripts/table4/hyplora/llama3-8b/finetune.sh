#!/bin/bash
python finetune_llama3.py \
    --base_model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --data_path './ft-training_set/commonsense_170k.json' \
    --output_dir './trained_models/llama3-8b-hyplora-k1.0-r32-commonsense' \
    --batch_size 16 --micro_batch_size 4 \
    --num_epochs 3 --learning_rate 3e-4 \
    --cutoff_len 256 --val_set_size 120 \
    --eval_step 80 --save_step 80 \
    --adapter_name lora --lora_r 32 --lora_alpha 128 \
    --target_modules '["q_proj", "v_proj", "k_proj", "up_proj", "down_proj"]' \
    --lora_type "hyplora-1.0"

#!/bin/bash
python finetune_qwen.py \
    --base_model "Qwen/Qwen2.5-7B-Instruct-1M" \
    --data_path './ft-training_set/commonsense_170k.json' \
    --output_dir './trained_models/qwen2.5-7b-lora-r32-commonsense' \
    --batch_size 16 --micro_batch_size 4 \
    --num_epochs 3 --learning_rate 3e-4 \
    --cutoff_len 256 --val_set_size 120 \
    --eval_step 80 --save_step 80 \
    --adapter_name lora --lora_r 32 --lora_alpha 64 \
    --target_modules '["q_proj", "v_proj", "k_proj", "up_proj", "down_proj"]' \
    --lora_type "std"

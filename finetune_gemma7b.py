import os
import sys
import random
from typing import List

import fire
import numpy as np
import torch
import transformers
from datasets import load_dataset

_hf_token = os.environ.get("HF_TOKEN")

from peft import (
    LoraConfig,
    get_peft_model
)
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def train(
        # model/data params
        base_model: str = "",
        data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./lora-alpaca",
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 256,
        val_set_size: int = 2000,
        use_gradient_checkpointing: bool = False,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_type: str = 'std',
        use_dora: bool=False,
        target_modules: List[str] = None,
        # llm hyperparams
        train_on_inputs: bool = True,
        group_by_length: bool = False,
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",
        wandb_log_model: str = "",
        resume_from_checkpoint: str = None,
):

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)

    print(
        f"Finetuning model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_type: {lora_type}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"use_dora: {use_dora}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"target_modules: {target_modules}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )

    assert (
        base_model
    ), "Please specify a --base_model"
    gradient_accumulation_steps = batch_size // micro_batch_size


    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )

    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        token=_hf_token,
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, token=_hf_token)
    tokenizer.padding_side = "right"

    def generate_prompt(data_point):
        instruction = str(data_point.get("instruction") or "").strip()
        input_text = str(data_point.get("input") or "").strip()
        output_text = str(data_point.get("output") or "").strip()

        if input_text:
            prompt = (
                "Below is an instruction that describes a task, paired "
                "with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                "### Response:\n"
            )
        else:
            prompt = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                "### Response:\n"
            )

 
        if output_text:
            prompt += output_text
            if tokenizer.eos_token:
                prompt += tokenizer.eos_token

        return prompt

    def tokenize(messages):
        result = tokenizer(
            messages,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )

        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)

        if not train_on_inputs:
            user_prompt = generate_prompt(
                {**data_point, "output": ""}
            )
            tokenized_user_prompt = tokenize(user_prompt)

            user_prompt_len = min(
                len(tokenized_user_prompt["input_ids"]),
                len(tokenized_full_prompt["labels"]),
            )

            tokenized_full_prompt["labels"] = (
                [-100] * user_prompt_len
                + tokenized_full_prompt["labels"][user_prompt_len:]
            )

        return tokenized_full_prompt
        
    

    config = LoraConfig(
                use_dora=use_dora,
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                lora_type=lora_type,
                bias="none",
                task_type="CAUSAL_LM",
            )
    model = get_peft_model(model, config)
    
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        
    model.config.use_cache = False

    if data_path.endswith(".json"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        if not os.path.isdir(resume_from_checkpoint):
            raise ValueError(f"Checkpoint directory not found: {resume_from_checkpoint}")
        print(f"Resuming training from {resume_from_checkpoint}")
    else:
        print("Starting fresh training run.")

    model.print_trainable_parameters()
    
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=seed
        )
        train_data = (
            train_val["train"].shuffle(seed=seed).map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle(seed=seed).map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle(seed=seed).map(generate_and_tokenize_prompt)
        val_data = None
    
    trainer = transformers.Trainer(
        model= model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=10,
            optim="adamw_torch_fused",
            eval_strategy="epoch",
            save_strategy="epoch",
            output_dir=output_dir,
            save_total_limit=3,
            label_names=["labels"],
            load_best_model_at_end=True if val_set_size > 0 else False,
            group_by_length=group_by_length,
            seed=seed,
            data_seed=seed,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    
    torch.cuda.reset_peak_memory_stats()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    print(f"Max Allocated Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"Max Reserved Memory: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)

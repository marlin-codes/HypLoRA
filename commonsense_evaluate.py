"""
Unified Commonsense Evaluation Script

Supports all models:
- LLaMA-7B, LLaMA-13B, BLOOM-7B, GPT-j-6B (legacy)
- Qwen2.5-7B-Instruct-1M, Qwen2.5-7B-Instruct
- gemma-3-1b-it, gemma-3-4b-it, gemma-7b, gemma-7b-it
- Meta-Llama-3-8B-Instruct
"""

import copy
import json
import os
import re
import sys
import argparse

import torch

sys.path.insert(0, os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    GenerationConfig,
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    Gemma3ForConditionalGeneration
)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


# =============================================================================
# Prompt Templates
# =============================================================================

def generate_prompt_legacy(instruction, input_text=None):
    """For LLaMA-7B, LLaMA-13B, BLOOM-7B, GPT-j-6B"""
    if input_text:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input_text}

                ### Response:
                """
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Response:
                """


def generate_prompt_qwen(instruction, input_text=None):
    """For Qwen2.5-7B-Instruct-1M, Qwen2.5-7B-Instruct"""
    system_prompt = "You are a helpful assistant."

    if input_text:
        user_message = f"{instruction.strip()}\n\nInput:\n{input_text.strip()}"
    else:
        user_message = instruction.strip()

    return f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""


def generate_prompt_gemma(instruction, input_text=None):
    """For gemma-3-4b-it, gemma-3-1b-it, gemma-7b (base)"""
    prompt = "<|system|>\nYou are a helpful assistant.\n"

    if input_text:
        prompt += "<|user|>\n" + instruction.strip() + "\n\n### Input:\n" + input_text.strip() + "\n"
    else:
        prompt += "<|user|>\n" + instruction.strip() + "\n"

    prompt += "<|assistant|>\n"
    return prompt


def generate_prompt_gemma7b_it(instruction, input_text=None):
    """For gemma-7b-it (instruction-tuned)"""
    if input_text:
        prompt = (
            f"<start_of_turn>user\n"
            f"You are a helpful assistant.\n\n{instruction.strip()}\n\nInput:\n{input_text.strip()}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
    else:
        prompt = (
            f"<start_of_turn>user\n"
            f"You are a helpful assistant.\n\n{instruction.strip()}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
    return prompt


def generate_prompt_llama3(instruction, input_text=None):
    """For Meta-Llama-3-8B-Instruct"""
    if input_text:
        formatted_prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    else:
        formatted_prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    return formatted_prompt


def generate_prompt(instruction, input_text=None, model_name="LLaMA-7B"):
    """Unified prompt generation based on model type"""
    if model_name in ("LLaMA-7B", "LLaMA-13B", "BLOOM-7B", "GPT-j-6B"):
        return generate_prompt_legacy(instruction, input_text)
    elif model_name in ("Qwen2.5-7B-Instruct-1M", "Qwen2.5-7B-Instruct", "qwen2.5-7b-instruct"):
        return generate_prompt_qwen(instruction, input_text)
    elif model_name in ("gemma-3-4b-it", "gemma-3-1b-it", "gemma-7b"):
        return generate_prompt_gemma(instruction, input_text)
    elif model_name == "gemma-7b-it":
        return generate_prompt_gemma7b_it(instruction, input_text)
    elif model_name in ("Meta-Llama-3-8B-Instruct", "llama-3-8b-instruct"):
        return generate_prompt_llama3(instruction, input_text)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# Output Parsing
# =============================================================================

def parse_output(output, model_name):
    """Parse model output based on model type"""
    try:
        if model_name in ("LLaMA-7B", "LLaMA-13B", "BLOOM-7B", "GPT-j-6B"):
            return output.split("### Response:")[1].strip()
        elif model_name in ("Qwen2.5-7B-Instruct-1M", "Qwen2.5-7B-Instruct", "qwen2.5-7b-instruct"):
            return output.split("assistant\n", 1)[1].strip()
        elif model_name in ("gemma-3-4b-it", "gemma-3-1b-it", "gemma-7b"):
            return output.split("<|assistant|>", 1)[1].split("<|end|>")[0].strip()
        elif model_name == "gemma-7b-it":
            if "<start_of_turn>model" in output:
                return output.split("<start_of_turn>model", 1)[1].split("<end_of_turn>")[0].strip()
            return output.strip()
        elif model_name in ("Meta-Llama-3-8B-Instruct", "llama-3-8b-instruct"):
            return output.split("assistant\n\n", 1)[1].strip()
        else:
            return output.strip()
    except (IndexError, AttributeError):
        return output.strip()


# =============================================================================
# Model Loading
# =============================================================================

def load_model(args):
    """Load model and tokenizer with proper configuration for each model type"""
    base_model = args.base_model
    if not base_model:
        raise ValueError(f'Cannot find base model name: {args.model}')

    lora_weights = args.lora_weights
    if not lora_weights:
        raise ValueError(f'Cannot find lora weight: {lora_weights}')

    load_8bit = args.load_8bit

    # Load tokenizer
    if "LLaMA" in args.model and args.model in ("LLaMA-7B", "LLaMA-13B"):
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Set padding for decoder-only models (CRITICAL for batch inference)
    decoder_only_models = (
        "Meta-Llama-3-8B-Instruct",
        "llama-3-8b-instruct",
        "Qwen2.5-7B-Instruct-1M",
        "Qwen2.5-7B-Instruct",
        "qwen2.5-7b-instruct",
        "gemma-3-1b-it",
        "gemma-3-4b-it",
        "gemma-7b",
        "gemma-7b-it"
    )
    if args.model in decoder_only_models:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    else:
        # Legacy models
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = 0  # unk

    # Load model
    if device == "cuda":
        if args.model in ("gemma-3-1b-it", "gemma-3-4b-it"):
            # Gemma 3 requires special handling
            model = Gemma3ForConditionalGeneration.from_pretrained(
                base_model,
                device_map={"": 0},
                attn_implementation='eager'
            )
            model = model.language_model
            model.config.use_cache = False
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={"": 0}
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

        # Fix for some legacy configs
        model.config.pad_token_id = tokenizer.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not load_8bit:
            model.half()

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

    return tokenizer, model


# =============================================================================
# Data Loading & Batching
# =============================================================================

def load_data(args):
    """Load test dataset"""
    file_path = f'dataset/{args.dataset}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find dataset file: {file_path}")
    return json.load(open(file_path, 'r'))


def create_batch(dataset, batch_size):
    """Create batches from dataset"""
    batches = []
    num_batch = len(dataset) // batch_size + (1 if len(dataset) % batch_size != 0 else 0)
    for i in range(num_batch):
        batch = dataset[i * batch_size: min((i + 1) * batch_size, len(dataset))]
        batches.append(batch)
    return batches


def create_dir(dir_path):
    """Create directory if not exists"""
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


# =============================================================================
# Answer Extraction
# =============================================================================

def extract_answer(args, sentence):
    """Extract answer from model output based on dataset type"""
    dataset = args.dataset
    sentence_ = sentence.strip().lower()

    if dataset == 'boolq':
        pred_answers = re.findall(r'true|false', sentence_)
        return pred_answers[0] if pred_answers else ""

    elif dataset == 'piqa':
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        return pred_answers[0] if pred_answers else ""

    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        return pred_answers[0] if pred_answers else ""

    elif dataset == 'hellaswag':
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        return pred_answers[0] if pred_answers else ""

    elif dataset == 'winogrande':
        pred_answers = re.findall(r'option1|option2', sentence_)
        return pred_answers[0] if pred_answers else ""

    return ""


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Commonsense Evaluation")

    parser.add_argument('--dataset',
                        choices=["boolq", "piqa", "social_i_qa", "hellaswag",
                                 "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"],
                        required=True,
                        help="Evaluation dataset")

    parser.add_argument('--model',
                        choices=['LLaMA-7B', 'LLaMA-13B', 'BLOOM-7B', 'GPT-j-6B',
                                 'Qwen2.5-7B-Instruct-1M', 'Qwen2.5-7B-Instruct',
                                 'qwen2.5-7b-instruct',  # alias
                                 'gemma-3-1b-it', 'gemma-3-4b-it', 'gemma-7b', 'gemma-7b-it',
                                 'Meta-Llama-3-8B-Instruct', 'llama-3-8b-instruct'],  # alias
                        required=True,
                        help="Model type")

    parser.add_argument('--adapter',
                        choices=['No', 'LoRA', 'AdapterP', 'AdapterH', 'Parallel'],
                        required=True,
                        help="Adapter type")

    parser.add_argument('--base_model', required=True, help="Base model path or name")
    parser.add_argument('--lora_weights', required=True, help="LoRA weights path")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size for evaluation")

    # Extended arguments (optional for legacy compatibility)
    parser.add_argument('--rank', type=int, default=32, help="LoRA rank")
    parser.add_argument('--lora_type', default='std', help="LoRA type")
    parser.add_argument('--lora_alpha', default='64', help="LoRA alpha")
    parser.add_argument('--use_dora', action="store_true", help="Use DoRA")
    parser.add_argument('--load_8bit', action='store_true', default=False, help="Load in 8bit")

    return parser.parse_args()


# =============================================================================
# Main Evaluation
# =============================================================================

def main():
    args = parse_args()

    # Load model and tokenizer
    tokenizer, model = load_model(args)

    # Setup save file
    if args.use_dora:
        save_file = f'experiment/{args.model}-dora-{args.dataset}-{args.rank}-{args.lora_type}-{args.lora_alpha}.json'
    else:
        save_file = f'experiment/{args.model}-{args.adapter}-{args.dataset}-{args.rank}-{args.lora_type}-{args.lora_alpha}.json'
    create_dir('experiment/')

    # Load data
    dataset = load_data(args)
    batches = create_batch(dataset, args.batch_size)

    def evaluate(instructions, input_text=None, num_beams=4, max_new_tokens=32):
        """Run evaluation on a batch of instructions"""
        prompts = [generate_prompt(inst, input_text, args.model) for inst in instructions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        inputs["input_ids"] = inputs["input_ids"].to(device)
        inputs["attention_mask"] = inputs["attention_mask"].to(device)

        generation_config = GenerationConfig(num_beams=num_beams)

        # Set generation config tokens
        generation_config.pad_token_id = tokenizer.pad_token_id
        generation_config.eos_token_id = tokenizer.eos_token_id
        generation_config.bos_token_id = tokenizer.bos_token_id

        # For Gemma models, set cache implementation
        if hasattr(model.config, 'cache_implementation'):
            generation_config.cache_implementation = model.config.cache_implementation

        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )

        sequences = generation_output.sequences
        outputs = tokenizer.batch_decode(sequences, skip_special_tokens=True)

        # Parse outputs based on model type
        outputs = [parse_output(o, args.model) for o in outputs]

        return outputs

    # Run evaluation
    total = len(batches)
    correct = 0
    current = 0
    output_data = []

    pbar = tqdm(total=total)
    for idx, batch in enumerate(batches):
        current += len(batch)
        instructions = [data.get('instruction') for data in batch]

        outputs = evaluate(instructions)

        for data, output in zip(batch, outputs):
            label = data.get('answer')
            predict = extract_answer(args, output)
            flag = (label == predict)

            if flag:
                correct += 1

            new_data = copy.deepcopy(data)
            new_data['output_pred'] = output
            new_data['pred'] = predict
            new_data['flag'] = flag
            new_data['accuracy'] = correct / current
            output_data.append(new_data)

            print(data["instruction"])
            print(output)
            print('prediction:', predict)
            print('label:', label)

        print('---------------')
        print(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / current}')
        print('---------------')

        with open(save_file, 'w+') as f:
            json.dump(output_data, f, indent=4)

        pbar.update(1)

    pbar.close()
    print('\n')
    print('test finished')
    print(f'Final accuracy: {correct / current:.4f}')


if __name__ == "__main__":
    main()

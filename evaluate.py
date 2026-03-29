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
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, Gemma3ForCausalLM, Gemma3ForConditionalGeneration

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def main(base_model: str = "", lora_weights: str = "tloen/alpaca-lora-7b"):
    args = parse_args()

    def evaluate(
            instruction,
            input=None,
            num_beams=4,
            max_new_tokens=512,
            **kwargs,
    ):
        prompt = generate_prompt(instruction, input, args.model)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        generation_config = GenerationConfig(
            num_beams=num_beams,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            **kwargs,
        )
        
        if args.model == "gemma-3-4b-it":
            generation_config.cache_implementation = model.config.cache_implementation
        
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s, skip_special_tokens=False)

        if args.model == "Meta-Llama-3-8B-Instruct":
            # ## for llama3-8B
            return output.split("<|start_header_id|>assistant<|end_header_id|>", 1)[1].split("<|eot_id|>", 1)[0].strip()
        elif args.model == "Qwen2.5-7B-Instruct-1M":
            # ## for qwen
            if "<|im_start|>assistant" in output:
                parts = output.split("<|im_start|>assistant", 1)
                if len(parts) > 1:
                    return parts[1].split("<|im_end|>", 1)[0].strip()
            return output.strip()
        elif args.model == "gemma-3-4b-it" or args.model == "gemma-7b":
            # for gemma
            return (output.split("<|assistant|>", 1)[1].split("<|end_of_turn|>", 1)[0].strip())
        else:
            return output

    def evaluate_batch(instructions, input=None, num_beams=4, max_new_tokens=512, **kwargs):
        """Batch evaluation of multiple instructions"""
        prompts = [generate_prompt(inst, input, args.model) for inst in instructions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

        generation_config = GenerationConfig(
            num_beams=num_beams,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            **kwargs,
        )

        if args.model == "gemma-3-4b-it":
            generation_config.cache_implementation = model.config.cache_implementation

        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens
            )

        outputs = []
        for s in generation_output.sequences:
            output = tokenizer.decode(s, skip_special_tokens=False)
            if args.model == "Meta-Llama-3-8B-Instruct":
                output = output.split("<|start_header_id|>assistant<|end_header_id|>", 1)[1].split("<|eot_id|>", 1)[0].strip()
            elif args.model == "Qwen2.5-7B-Instruct-1M":
                if "<|im_start|>assistant" in output:
                    parts = output.split("<|im_start|>assistant", 1)
                    if len(parts) > 1:
                        output = parts[1].split("<|im_end|>", 1)[0].strip()
                else:
                    output = output.strip()
            elif args.model == "gemma-3-4b-it" or args.model == "gemma-7b":
                output = output.split("<|assistant|>", 1)[1].split("<|end_of_turn|>", 1)[0].strip()
            outputs.append(output)
        return outputs

    if args.use_dora:
        save_file = f'experiment/{args.model}-dora-{args.dataset}-{args.rank}-{args.lora_type}-{args.lora_alpha}.json'
    else:
        save_file = f'experiment/{args.model}-{args.adapter}-{args.dataset}-{args.rank}-{args.lora_type}-{args.lora_alpha}-{args.run}.json'

    def flush_outputs():
        with open(save_file, 'w+') as f:
            json.dump(output_data, f, indent=4)

    create_dir('experiment/')

    dataset = load_data(args)
    tokenizer, model = load_model(args)
    total = len(dataset)
    correct = 0
    miss = 0.001
    output_data = []
    pbar = tqdm(total=total)

    batch_size = args.batch_size

    if batch_size > 1:
        # Batch evaluation
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_data = dataset[batch_start:batch_end]
            batch_instructions = [d.get('instruction') for d in batch_data]

            batch_outputs = evaluate_batch(batch_instructions)

            for i, (data, outputs) in enumerate(zip(batch_data, batch_outputs)):
                idx = batch_start + i
                label = data.get('answer')
                flag = False
                if args.dataset.lower() in ['aqua']:
                    predict = extract_answer_letter(args, outputs)
                    if label == predict:
                        correct += 1
                        flag = True
                else:
                    if isinstance(label, str):
                        label = float(label)
                    predict = extract_answer_number(args, outputs)
                    if abs(label - predict) <= miss:
                        correct += 1
                        flag = True
                new_data = copy.deepcopy(data)
                new_data['output_pred'] = outputs
                new_data['pred'] = predict
                new_data['flag'] = flag
                new_data['accuracy'] = correct / (idx + 1)
                output_data.append(new_data)
                print(' ')
                print('---------------')
                print(outputs)
                print('prediction:', predict)
                print('label:', label)
                print('---------------')
                print(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / (idx + 1)}')
                pbar.update(1)
            flush_outputs()
    else:
        # Single sample evaluation
        for idx, data in enumerate(dataset):
            instruction = data.get('instruction')

            outputs = evaluate(instruction)
            label = data.get('answer')
            flag = False
            if args.dataset.lower() in ['aqua']:
                predict = extract_answer_letter(args, outputs)
                if label == predict:
                    correct += 1
                    flag = True
            else:
                if isinstance(label, str):
                    label = float(label)
                predict = extract_answer_number(args, outputs)
                if abs(label - predict) <= miss:
                    correct += 1
                    flag = True
            new_data = copy.deepcopy(data)
            new_data['output_pred'] = outputs
            new_data['pred'] = predict
            new_data['flag'] = flag
            new_data['accuracy'] = correct / (idx + 1)
            output_data.append(new_data)
            print(' ')
            print('---------------')
            print(outputs)
            print('prediction:', predict)
            print('label:', label)
            print('---------------')
            print(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / (idx + 1)}')
            flush_outputs()
            pbar.update(1)
    pbar.close()
    flush_outputs()
    print('\n')
    print('test finished')


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return

# for all
def generate_prompt(instruction, input_text=None, llm_base="Qwen2.5-7B-Instruct-1M"):
    if llm_base == "Qwen2.5-7B-Instruct-1M":
        print("In llm base qwen")
        system_prompt = "You are a helpful assistant."
        if input_text:
            user_message = f"{instruction.strip()}\n\nInput:\n{input.strip()}"
        else:
            user_message = instruction.strip()

        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_message}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt
    elif llm_base == "gemma-3-4b-it" or llm_base == "gemma-7b":
        print(f"In llm base {llm_base}")
        prompt = "<|system|>\nYou are a helpful assistant.\n<|end_of_turn|>\n"
        
        if input_text:
            prompt += (
                "<|user|>\n"
                f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction.strip()}\n\n"
                f"### Input:\n{input_text.strip()}\n"
                "<|end_of_turn|>\n"
            )
        else:
            prompt += (
                "<|user|>\n"
                f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction.strip()}\n"
                "<|end_of_turn|>\n"
            )
        
        prompt += "<|assistant|>\n"
        return prompt
    
    elif llm_base == "Meta-Llama-3-8B-Instruct":
        print("In llm base llama3-8B")
        if input_text:
            formatted_prompt = (
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
                f"<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        else:
            formatted_prompt = (
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
                f"<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        return formatted_prompt


def load_data(args) -> list:
    file_path = f'dataset/{args.dataset}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['AddSub', 'MultiArith', 'SingleEq', 'gsm8k', 'AQuA', 'SVAMP', 'mawps'],
                        required=True)
    parser.add_argument('--model', choices=['LLaMA-7B', 'Qwen2.5-7B-Instruct-1M', 'gemma-3-4b-it', 'gemma-7b', 'Meta-Llama-3-8B-Instruct'], required=True)
    parser.add_argument('--adapter', choices=['No', 'LoRA'], required=True)
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--lora_weights', required=True)
    parser.add_argument('--lora_type', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--lora_alpha', required=True)
    parser.add_argument('--use_dora', action="store_true")
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')

    return parser.parse_args()


def load_model(args) -> tuple:
    base_model = args.base_model
    lora_weights = args.lora_weights

    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args.model}')
    if not lora_weights:
        raise ValueError(f'can not find lora weight, the value is: {lora_weights}')

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Set pad token if not set
    if args.model == "Meta-Llama-3-8B-Instruct":
        tokenizer.pad_token = tokenizer.eos_token

    # Use left padding for batch inference (decoder-only models)
    if args.batch_size > 1:
        tokenizer.padding_side = "left"
    else:
        tokenizer.padding_side = "right"

    if device == "cuda":
        if args.model == "gemma-3-4b-it":
            # model = Gemma3ForCausalLM.from_pretrained(base_model,torch_dtype=torch.float16, device_map={"": 0}, attn_implementation='eager')
            model = Gemma3ForConditionalGeneration.from_pretrained(base_model, device_map={"": 0}, attn_implementation='eager')
            model = model.language_model
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=False,
                torch_dtype=torch.float16,
                device_map={"": 0},
                trust_remote_code=True,
            )

        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={"": 0}
        )
        
    else:
        if args.model == "gemma-3-4b-it":
            model = Gemma3ForConditionalGeneration.from_pretrained(base_model, device_map={"": device}, attn_implementation='eager')
            model = model.language_model
        else:
            model = AutoModelForCausalLM.from_pretrained(base_model, device_map={"": device})

        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    return tokenizer, model


def load_instruction(args) -> str:
    instruction = ''
    if not instruction:
        raise ValueError('instruct not initialized')
    return instruction


def extract_answer_number(args, sentence: str) -> float:
    dataset = args.dataset.lower()
    if dataset in ["multiarith", "addsub", "singleeq", "gsm8k", "svamp", "mawps"]:
        sentence = sentence.replace(',', '')
        pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
        if not pred:
            return float('inf')
        pred_answer = float(pred[-1])
    else:
        raise NotImplementedError(' not support dataset: {}'.format(dataset))
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer

def extract_answer_letter(args, sentence: str) -> str:
    sentence = sentence.strip()

    # Original extraction method (before Feb 4, 2026):
    # matches = re.findall(r'\(?([A-E])\)', sentence)
    # if not matches:
    #     matches = re.findall(r'correct answer is\s+([A-E])', sentence, re.IGNORECASE)
    # if matches:
    #     return matches[-1]
    # return 'N/A'

    # Updated extraction method - handles LaTeX \boxed{} notation from HyperLoRA outputs
    # Try multiple patterns in order of specificity

    # Pattern 1: LaTeX \boxed{A} or \boxed{\text{A}} or \boxed{\text{(A)}}
    matches = re.findall(r'\\boxed\{\\text\{\(?([A-E])\)?\}\}', sentence)
    if not matches:
        matches = re.findall(r'\\boxed\{\(?([A-E])\)?\}', sentence)

    # Pattern 2: Standard parentheses (A), (B), etc.
    if not matches:
        matches = re.findall(r'\(?([A-E])\)', sentence)

    # Pattern 3: "correct answer is A" or "answer is (A)"
    if not matches:
        matches = re.findall(r'(?:correct )?answer is[:\s]+\(?([A-E])\)?', sentence, re.IGNORECASE)

    # Pattern 4: "The answer is A" at end of sentence
    if not matches:
        matches = re.findall(r'[Tt]he answer is[:\s]+\(?([A-E])\)?', sentence)

    # Pattern 5: Standalone letter at end after "is" (e.g., "Therefore, the answer is A.")
    if not matches:
        matches = re.findall(r'is[:\s]+([A-E])\.?\s*$', sentence)

    # Pattern 6: Option format "Option (A)" or "option A"
    if not matches:
        matches = re.findall(r'[Oo]ption\s+\(?([A-E])\)?', sentence)

    if matches:
        return matches[-1]
    return 'N/A'


if __name__ == "__main__":
    main()

import copy
import json
import os
import re
import sys
import argparse

import fire

import torch

sys.path.insert(0, os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, Gemma3ForConditionalGeneration

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def main(base_model: str = "", lora_weights: str = "tloen/alpaca-lora-7b"):
    args = parse_args()

    def evaluate(
            instructions,
            input=None,
            num_beams=4,
            max_new_tokens=32,
            **kwargs,
    ):
        if args.model == "gemma-3-4b-it" or args.model == "gemma-7b":
            prompts = [generate_prompt_gemma(instruction, input) for instruction in instructions]
        elif args.model == "Meta-Llama-3-8B-Instruct":
            prompts = [generate_prompt_llama(instruction, input) for instruction in instructions]
        else:
            print("Wrong choice of model selected")
            raise

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

        generation_config = GenerationConfig(
            num_beams=num_beams,
            pad_token_id = tokenizer.pad_token_id,
            eos_token_id = tokenizer.eos_token_id,
            bos_token_id = tokenizer.bos_token_id,
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
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
        if args.model == "gemma-3-4b-it" or args.model == "gemma-7b":
            ## for gemma
            outputs = [o.split("<|assistant|>", 1)[1].split("<|end_of_turn|>")[0].strip() for o in outputs]
        else:
            ## for llama 3 8b
            # outputs = [o.split("assistant\n\n", 1)[1].strip() for o in outputs]
            outputs = [o.split("assistant", 1)[1].strip() for o in outputs]

        return outputs

    if args.use_dora:
        save_file = f'experiment/{args.model}-dora-{args.dataset}-{args.rank}-{args.lora_type}-{args.lora_alpha}.json'
    else:
        save_file = f'experiment/{args.model}-{args.adapter}-{args.dataset}-{args.rank}-{args.lora_type}-{args.lora_alpha}.json'
    create_dir('experiment/')

    dataset = load_data(args)
    batches = create_batch(dataset, args.batch_size)
    tokenizer, model = load_model(args)
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
            flag = False
            predict = extract_answer(args, output)
            if label == predict:
                correct += 1
                flag = True
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


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return

# for gemma
def generate_prompt_gemma(instruction, input_text=None):
    prompt = "<|system|>\nYou are a helpful assistant.\n<|end_of_turn|>\n"

    if input_text:
        prompt += f"<|user|>\n{instruction.strip()}\n\nInput:\n{input_text.strip()}\n<|end_of_turn|>\n"
    else:
        prompt += f"<|user|>\n{instruction.strip()}\n<|end_of_turn|>\n"

    prompt += "<|assistant|>\n"
    return prompt

##for llama 3 b
def generate_prompt_llama(instruction, input_text=None):
    if input_text:
        formatted_prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
            f"<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    else:
        formatted_prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
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

def create_batch(dataset, batch_size):
    batches = []
    num_batch = len(dataset)//batch_size if len(dataset) % batch_size == 0 else len(dataset)//batch_size + 1
    for i in range(num_batch):
        batch = dataset[i*batch_size: min((i+1)*batch_size, len(dataset))]
        batches.append(batch)
    return batches


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"],
                        required=True)
    parser.add_argument('--model', choices=['LLaMA-7B', "LLaMA-13B",'BLOOM-7B', 'GPT-j-6B', 'Qwen2.5-7B-Instruct-1M', 'gemma-3-1b-it', 'gemma-3-4b-it', 'Meta-Llama-3-8B-Instruct', 'gemma-7b'], required=True)
    parser.add_argument('--adapter', choices=['No', 'LoRA', 'AdapterP', 'AdapterH', 'Parallel'],
                        required=True)
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--lora_weights', required=True)
    parser.add_argument('--lora_type', required=True)
    parser.add_argument('--lora_alpha', required=True)
    parser.add_argument('--use_dora', action="store_true")
    parser.add_argument('--batch_size', type=int, required=True)

    return parser.parse_args()


def load_model(args) -> tuple:
    base_model = args.base_model
    lora_weights = args.lora_weights

    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args.model}')
    if not lora_weights:
        raise ValueError(f'can not find lora weight, the value is: {lora_weights}')

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    if args.model == "Meta-Llama-3-8B-Instruct" or args.model == "gemma-3-4b-it" or args.model == "gemma-7b":
        if args.model == "Meta-Llama-3-8B-Instruct":
            tokenizer.pad_token = tokenizer.eos_token
        # Use left padding for generation (required for decoder-only models in batch mode)
        tokenizer.padding_side = "left"

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
                # device_map="auto",
                trust_remote_code=True,
            ) 
            
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={"":0}
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


def extract_answer(args, sentence: str) -> float:
    dataset = args.dataset
    if dataset == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'piqa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'hellaswag':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'winogrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]


if __name__ == "__main__":
    main()

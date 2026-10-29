import copy
import json
import os
import re
import sys
import argparse
from packaging import version
import random
import numpy as np
from contextlib import contextmanager


import torch
import transformers
from tqdm import tqdm
from transformers import (
    GenerationConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Gemma3ForCausalLM,
)
from peft import PeftModel

_hf_token = os.environ.get("HF_TOKEN")


SUPPORTED_MODELS = {
    "Meta-Llama-3-8B-Instruct",
    "Qwen2.5-7B-Instruct-1M",
    "gemma-3-4b-it",
    "gemma-7b",
}


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
transformers.set_seed(seed)

if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
else:
    device = "cpu"


def main(base_model: str = "", lora_weights: str = "tloen/alpaca-lora-7b"):
    args = parse_args()

    def format_eval_prompt(prompt):
        if args.model == "gemma-3-4b-it":
            return tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )

        return prompt

    def build_generation_config(num_beams=4, **kwargs):
        common = dict(
            num_beams=num_beams,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        if args.model in {"Meta-Llama-3-8B-Instruct", "gemma-7b"}:
            common["bos_token_id"] = tokenizer.bos_token_id

        elif args.model == "gemma-3-4b-it":
            common["bos_token_id"] = tokenizer.bos_token_id

            end_of_turn_id = tokenizer.convert_tokens_to_ids(
                "<end_of_turn>"
            )

            common["eos_token_id"] = [
                tokenizer.eos_token_id,
                end_of_turn_id,
            ]

            common["cache_implementation"] = "hybrid"

        common.update(kwargs)
        return GenerationConfig(**common)

    def decode_generated(generation_output, input_length):
        """
        Decode only newly generated tokens, not the prompt.

        For batch generation with left padding, all input sequences have the
        same padded length, so input_length is the common sequence length.
        """
        outputs = []

        for sequence in generation_output.sequences:
            generated_tokens = sequence[input_length:]
            output = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()
            outputs.append(output)

        return outputs

    def evaluate(
        instruction,
        input=None,
        num_beams=4,
        max_new_tokens=512,
        **kwargs,
    ):
        prompt = generate_prompt(instruction, input, args.model)
        formatted_prompt = format_eval_prompt(prompt)

        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=(args.model == "gemma-7b"),
        ).to(device)

        generation_config = build_generation_config(
            num_beams=num_beams,
            **kwargs,
        )

        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                use_model_defaults=False,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )

        input_length = inputs["input_ids"].shape[1]
        return decode_generated(generation_output, input_length)[0]

    def evaluate_batch(
        instructions,
        input_texts=None,
        num_beams=4,
        max_new_tokens=512,
        **kwargs,
    ):
        if input_texts is None:
            input_texts = [None] * len(instructions)

        if len(instructions) != len(input_texts):
            raise ValueError(
                f"instructions and input_texts must have the same length, "
                f"got {len(instructions)} and {len(input_texts)}"
            )

        prompts = [
            generate_prompt(inst, input_text, args.model)
            for inst, input_text in zip(instructions, input_texts)
        ]

        formatted_prompts = [
            format_eval_prompt(prompt)
            for prompt in prompts
        ]

        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=(args.model == "gemma-7b"),
        ).to(device)

        generation_config = build_generation_config(
            num_beams=num_beams,
            **kwargs,
        )

        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                use_model_defaults=False,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )

        input_length = inputs["input_ids"].shape[1]
        return decode_generated(generation_output, input_length)

    if args.use_dora:
        save_file = (
            f"experiment/{args.model}-dora-{args.dataset}-"
            f"{args.rank}-{args.lora_type}-{args.lora_alpha}.json"
        )
    else:
        save_file = (
            f"experiment/{args.model}-{args.adapter}-{args.dataset}-"
            f"{args.rank}-{args.lora_type}-{args.lora_alpha}-{args.run}.json"
        )

    create_dir("experiment/")

    dataset = load_data(args)
    tokenizer, model = load_model(args)

    total = len(dataset)
    correct = 0
    miss = 0.001
    output_data = []

    pbar = tqdm(total=total)
    batch_size = args.batch_size

    def flush_outputs():
        with open(save_file, "w") as f:
            json.dump(output_data, f, indent=4)

    if batch_size > 1:
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_data = dataset[batch_start:batch_end]

            batch_instructions = [
                d.get("instruction", "")
                for d in batch_data
            ]
            batch_inputs = [
                d.get("input")
                for d in batch_data
            ]

            batch_outputs = evaluate_batch(
                batch_instructions,
                batch_inputs,
            )

            for i, (data, output) in enumerate(
                zip(batch_data, batch_outputs)
            ):
                idx = batch_start + i
                label = data.get("answer")
                flag = False

                if args.dataset.lower() in ["aqua"]:
                    predict = extract_answer_letter(args, output)

                    if label == predict:
                        correct += 1
                        flag = True
                else:
                    if isinstance(label, str):
                        label = float(label)

                    predict = extract_answer_number(args, output)

                    if abs(label - predict) <= miss:
                        correct += 1
                        flag = True

                new_data = copy.deepcopy(data)
                new_data["output_pred"] = output
                new_data["pred"] = predict
                new_data["flag"] = flag
                new_data["accuracy"] = correct / (idx + 1)
                output_data.append(new_data)

                print()
                print("---------------")
                print(output)
                print("prediction:", predict)
                print("label:", label)
                print("---------------")
                print(
                    f"\rtest:{idx + 1}/{total} | "
                    f"accuracy {correct}  {correct / (idx + 1)}"
                )

                pbar.update(1)

            flush_outputs()

    else:
        for idx, data in enumerate(dataset):
            instruction = data.get("instruction", "")
            input_text = data.get("input")

            output = evaluate(
                instruction,
                input_text,
            )

            label = data.get("answer")
            flag = False

            if args.dataset.lower() in ["aqua"]:
                predict = extract_answer_letter(args, output)

                if label == predict:
                    correct += 1
                    flag = True
            else:
                if isinstance(label, str):
                    label = float(label)

                predict = extract_answer_number(args, output)

                if abs(label - predict) <= miss:
                    correct += 1
                    flag = True

            new_data = copy.deepcopy(data)
            new_data["output_pred"] = output
            new_data["pred"] = predict
            new_data["flag"] = flag
            new_data["accuracy"] = correct / (idx + 1)
            output_data.append(new_data)

            print()
            print("---------------")
            print(output)
            print("prediction:", predict)
            print("label:", label)
            print("---------------")
            print(
                f"\rtest:{idx + 1}/{total} | "
                f"accuracy {correct}  {correct / (idx + 1)}"
            )

            flush_outputs()
            pbar.update(1)

    pbar.close()
    flush_outputs()

    print("\n")
    print("test finished")


def create_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def generate_prompt(
    instruction,
    input_text=None,
    llm_base="Qwen2.5-7B-Instruct-1M",
):
    if llm_base == "Qwen2.5-7B-Instruct-1M":
        system_prompt = "You are a helpful assistant."

        if input_text:
            user_message = (
                f"{instruction.strip()}\n\n"
                f"Input:\n{input_text.strip()}"
            )
        else:
            user_message = instruction.strip()

        return (
            f"<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{user_message}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
    elif llm_base == "gemma-7b":

        if input_text:
            return (
                "Below is an instruction that describes a task, paired "
                "with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction.strip()}\n\n"
                f"### Input:\n{input_text.strip()}\n\n"
                "### Response:\n"
            )

        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction.strip()}\n\n"
            "### Response:\n"
        )

    elif llm_base == "gemma-3-4b-it":
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            }
        ]

        if input_text:
            user_message = (
                f"{instruction.strip()}\n\n"
                f"Input:\n{input_text.strip()}"
            )
        else:
            user_message = instruction.strip()

        messages.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        return messages

    elif llm_base == "Meta-Llama-3-8B-Instruct":
        if input_text:
            return (
                "<|begin_of_text|>"
                "<|start_header_id|>user<|end_header_id|>\n\n"
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                "### Response:\n"
                "<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )

        return (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            "### Response:\n"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    else:
        raise ValueError(f"Unsupported model: {llm_base}")


def load_data(args):
    file_path = f"dataset/{args.dataset}/test.json"

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"can not find dataset file: {file_path}"
        )

    with open(file_path, "r") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=[
            "AddSub",
            "MultiArith",
            "SingleEq",
            "gsm8k",
            "AQuA",
            "SVAMP",
            "mawps",
        ],
        required=True,
    )

    parser.add_argument(
        "--model",
        choices=[
            "Meta-Llama-3-8B-Instruct",
            "Qwen2.5-7B-Instruct-1M",
            "gemma-3-4b-it",
            "gemma-7b",
        ],
        required=True,
    )

    parser.add_argument(
        "--adapter",
        choices=["No", "LoRA"],
        required=True,
    )
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--lora_weights", required=True)
    parser.add_argument("--lora_type", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--lora_alpha", required=True)
    parser.add_argument("--use_dora", action="store_true")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference",
    )

    return parser.parse_args()


def load_model(args):
    base_model = args.base_model
    lora_weights = args.lora_weights

    if not base_model:
        raise ValueError(
            f"can not find base model name by the value: {args.model}"
        )

    if not lora_weights:
        raise ValueError(
            f"can not find lora weight, the value is: {lora_weights}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        token=_hf_token,
    )

    if args.model == "Meta-Llama-3-8B-Instruct":
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == "gemma-7b" and tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.batch_size > 1:
        tokenizer.padding_side = "left"
    else:
        tokenizer.padding_side = "right"

    if device == "cuda":
        if args.model == "gemma-3-4b-it":
            model = Gemma3ForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
                trust_remote_code=True,
                token=_hf_token,
                attn_implementation="eager",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=False,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
                trust_remote_code=True,
                token=_hf_token,
                attn_implementation="flash_attention_2",
            )
    else:
        if args.model == "gemma-3-4b-it":
            model = Gemma3ForCausalLM.from_pretrained(
                base_model,
                device_map={"": device},
                trust_remote_code=True,
                token=_hf_token,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map={"": device},
                trust_remote_code=True,
                token=_hf_token,
            )

    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.bfloat16 if device == "cuda" else None,
        device_map={"": 0 if device == "cuda" else device},
    )

    model.eval()

    return tokenizer, model


def extract_answer_number(args, sentence: str) -> float:
    dataset = args.dataset.lower()

    if dataset in [
        "multiarith",
        "addsub",
        "singleeq",
        "gsm8k",
        "svamp",
        "mawps",
    ]:
        sentence = sentence.replace(",", "")
        pred = re.findall(r"-?\d+\.?\d*", sentence)

        if not pred:
            return float("inf")

        pred_answer = float(pred[-1])
    else:
        raise NotImplementedError(
            f"not support dataset: {dataset}"
        )

    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError:
            pred_answer = float("inf")

    return pred_answer


def extract_answer_letter(args, sentence: str) -> str:
    sentence = sentence.strip()

    matches = re.findall(
        r"\\boxed\{\\text\{\(?([A-E])\)?\}\}",
        sentence,
    )

    if not matches:
        matches = re.findall(
            r"\\boxed\{\(?([A-E])\)?\}",
            sentence,
        )

    if not matches:
        matches = re.findall(
            r"\(?([A-E])\)",
            sentence,
        )

    if not matches:
        matches = re.findall(
            r"(?:correct )?answer is[:\s]+\(?([A-E])\)?",
            sentence,
            re.IGNORECASE,
        )

    if not matches:
        matches = re.findall(
            r"[Tt]he answer is[:\s]+\(?([A-E])\)?",
            sentence,
        )

    if not matches:
        matches = re.findall(
            r"is[:\s]+([A-E])\.?\s*$",
            sentence,
        )

    if not matches:
        matches = re.findall(
            r"[Oo]ption\s+\(?([A-E])\)?",
            sentence,
        )

    if matches:
        return matches[-1]

    return "N/A"


if __name__ == "__main__":
    main()
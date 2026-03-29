"""
Reproduce Table 2: delta-hyperbolicity across all models and datasets.

Usage:
    # Run all combinations (4 models x 4 datasets = 16 runs, sequential on 1 GPU)
    CUDA_VISIBLE_DEVICES=0 python utils/run_delta_all.py

    # Or specify a single model
    CUDA_VISIBLE_DEVICES=0 python utils/run_delta_all.py --model gemma-7b
"""

import os
import sys
import json
import torch
import numpy as np
from scipy.spatial import distance_matrix
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse


MODEL_CONFIGS = {
    "LLaMA-7B": {
        "base_model": "huggyllama/llama-7b",
    },
    "LLaMA-13B": {
        "base_model": "huggyllama/llama-13b",
    },
    "Gemma-7B": {
        "base_model": "google/gemma-7b",
    },
    "LLaMA3-8B": {
        "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
    },
}

DATASETS = ["mawps", "SVAMP", "gsm8k", "AQuA"]


def delta_hyp(dismat):
    p = 0
    row = dismat[p, :][np.newaxis, :]
    col = dismat[:, p][:, np.newaxis]
    XY_p = 0.5 * (row + col - dismat)
    maxmin = np.max(np.minimum(XY_p[:, :, None], XY_p[None, :, :]), axis=1)
    return np.max(maxmin - XY_p)


def get_delta(output_features, max_points=1500):
    all_features = output_features.squeeze().cpu().detach().numpy()
    if len(all_features) <= max_points:
        all_features_small = all_features
    else:
        idx = np.random.choice(len(all_features), max_points, replace=False)
        all_features_small = all_features[idx]

    dists = distance_matrix(all_features_small, all_features_small)
    delta = delta_hyp(dists)
    diam = np.max(dists)
    return delta, diam


def run_model_dataset(model, tokenizer, dataset, device):
    """Compute delta-hyperbolicity for a single model-dataset pair."""
    delta_ratios = []
    for data in tqdm(dataset, desc="Computing Delta", leave=False):
        instruction = data.get('instruction')
        input_ids = tokenizer(instruction, return_tensors="pt")["input_ids"].to(device)
        input_embeddings = model.get_input_embeddings()(input_ids)
        delta, diam = get_delta(input_embeddings)
        delta_ratios.append(2 * delta / diam if diam > 0 else 0)
    return np.mean(delta_ratios), np.std(delta_ratios)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str,
                        help='Run a single model (e.g., Gemma-7B). Default: all models.')
    parser.add_argument('--dataset_root', default='dataset', type=str)
    parser.add_argument('--max_samples', default=0, type=int,
                        help='Max samples per dataset (0=all)')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    models_to_run = {args.model: MODEL_CONFIGS[args.model]} if args.model else MODEL_CONFIGS

    # results[model_name][dataset_name] = (mean, std)
    results = {}

    for model_name, config in models_to_run.items():
        print(f"\n{'='*60}")
        print(f"Loading {model_name} ({config['base_model']})")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(config["base_model"], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            config["base_model"],
            torch_dtype=torch.float16,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            trust_remote_code=True,
        )

        results[model_name] = {}

        for ds_name in DATASETS:
            ds_path = f'{args.dataset_root}/{ds_name}/test.json'
            if not os.path.exists(ds_path):
                print(f"  [SKIP] {ds_name}: {ds_path} not found")
                continue

            dataset = json.load(open(ds_path, 'r'))
            if args.max_samples > 0:
                dataset = dataset[:args.max_samples]

            print(f"  {ds_name} ({len(dataset)} samples)...", end=" ", flush=True)
            mean, std = run_model_dataset(model, tokenizer, dataset, device)
            results[model_name][ds_name] = (mean, std)
            print(f"delta = {mean:.2f} +/- {std:.2f}")

        # Free GPU memory before loading next model
        del model, tokenizer
        torch.cuda.empty_cache()

    # Print Table 2 (right side)
    print(f"\n{'='*80}")
    print(f"Table 2: delta-Hyperbolicity across models and datasets")
    print(f"{'='*80}")

    header = f"{'Model':<15}"
    for ds in DATASETS:
        header += f"  {ds:>12}"
    print(header)
    print("-" * len(header))

    all_means = {ds: [] for ds in DATASETS}

    for model_name in results:
        row = f"{model_name:<15}"
        for ds in DATASETS:
            if ds in results[model_name]:
                mean, std = results[model_name][ds]
                row += f"  {mean:.2f} +/- {std:.2f}"
                all_means[ds].append(mean)
            else:
                row += f"  {'N/A':>12}"
        print(row)

    # Average row
    row = f"{'Average':<15}"
    for ds in DATASETS:
        if all_means[ds]:
            avg = np.mean(all_means[ds])
            std = np.std(all_means[ds])
            row += f"  {avg:.2f} +/- {std:.2f}"
        else:
            row += f"  {'N/A':>12}"
    print("-" * len(header))
    print(row)

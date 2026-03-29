"""
Compute Gromov delta-hyperbolicity of token embeddings.

Usage:
    python utils/delta_hyperbolicity.py \
        --dataset gsm8k \
        --model gemma-7b \
        --base_model google/gemma-7b \
        --output_folder utils/figs/delta
"""

import argparse
import os
import json
import torch
import numpy as np
from scipy.spatial import distance_matrix
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt


def delta_hyp(dismat):
    """
    Compute the Gromov delta-hyperbolicity of a distance matrix.
    Uses the 4-point condition with base point p=0.
    """
    p = 0
    row = dismat[p, :][np.newaxis, :]
    col = dismat[:, p][:, np.newaxis]
    XY_p = 0.5 * (row + col - dismat)
    maxmin = np.max(np.minimum(XY_p[:, :, None], XY_p[None, :, :]), axis=1)
    return np.max(maxmin - XY_p)


def get_delta(output_features, max_points=1500):
    """
    Compute delta-hyperbolicity from feature vectors.
    Subsamples to max_points if needed for efficiency.
    """
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


def plot_delta_distribution(delta_ratios, output_folder, dataset_name):
    """Plot histogram of delta-hyperbolicity values."""
    os.makedirs(output_folder, exist_ok=True)

    plt.figure(figsize=(5, 4))
    n, bins, patches = plt.hist(delta_ratios, bins=30, alpha=0.7)

    cmap = plt.cm.coolwarm
    norm = plt.Normalize(0, 0.6)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for bin_center, patch in zip(bin_centers, patches):
        patch.set_facecolor(cmap(norm(bin_center)))

    mean_val = np.mean(delta_ratios)
    std_val = np.std(delta_ratios)
    plt.text(0.95, 0.95, f'Mean: {mean_val:.2f}$\\pm${std_val:.2f}',
             transform=plt.gca().transAxes, fontsize=18,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle="round", edgecolor='black', facecolor='none', alpha=0.5))

    plt.xlabel("Hyperbolicity ($\\delta$)", fontsize=18)
    plt.ylabel("Frequency", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    for ext in ['pdf', 'png']:
        path = os.path.join(output_folder, f"{dataset_name}_delta_hyperbolicity.{ext}")
        plt.savefig(path, bbox_inches='tight', dpi=300)
        print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--dataset_root', default='dataset', type=str)
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--base_model', required=True, type=str)
    parser.add_argument('--output_folder', default='utils/figs/delta', type=str)
    parser.add_argument('--max_samples', default=0, type=int,
                        help='Max samples to process (0=all)')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_path = f'{args.dataset_root}/{args.dataset}/test.json'
    print(f'Loading data from {dataset_path}')
    dataset = json.load(open(dataset_path, 'r'))
    if args.max_samples > 0:
        dataset = dataset[:args.max_samples]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
        trust_remote_code=True,
    )

    delta_ratios = []
    for data in tqdm(dataset, desc="Computing Delta"):
        instruction = data.get('instruction')
        input_ids = tokenizer(instruction, return_tensors="pt")["input_ids"].to(device)
        input_embeddings = model.get_input_embeddings()(input_ids)
        delta, diam = get_delta(input_embeddings)
        delta_ratios.append(2 * delta / diam if diam > 0 else 0)

    out_dir = f'{args.output_folder}/{args.model}'
    plot_delta_distribution(delta_ratios, out_dir, args.dataset)

    print(f"\n{args.model} on {args.dataset}: delta = {np.mean(delta_ratios):.4f} +/- {np.std(delta_ratios):.4f}")

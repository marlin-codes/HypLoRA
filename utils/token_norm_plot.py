import argparse
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer, Gemma3ForConditionalGeneration, Gemma3ForCausalLM
import os
import json
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.font_manager import FontProperties
import sys
from matplotlib import cm  # For colormap
import csv
import re

# make directory
def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    return dir_path


def load_data(args) -> list:
    # if not args.dataset.startswith("math"):
    #     file_path = f'{args.dataset_root}/{args.dataset}/test.json'
    # else:
    #     file_path = f'../ft-training_set/{args.dataset}.json'
    # file_path = f'./dataset/{args.dataset}/test.json'
    file_path = f'./ft-training_set/{args.dataset}.json'
    print(f'evaluate data from {file_path}')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data


def compute_norms(input_embeddings):
    
    norms = torch.norm(input_embeddings, dim=-1).detach().cpu().numpy()
    return norms.flatten()

def clean_token(token):
    # Keep alphanumeric and certain symbols like '$', '#'
    token = re.sub(r'^[^a-zA-Z0-9$#]+', '', token)  # Remove unwanted leading characters except for $, #
    token = re.sub(r'[^a-zA-Z0-9$#]+$', '', token)  # Remove unwanted trailing characters except for $, #
    return token.strip()

# Function to save token statistics to CSV sorted by frequency
def save_token_statistics_to_csv(token_frequency, token_norms, output_folder, dataset_name):
    # Create the CSV file path
    csv_file_path = os.path.join(output_folder, f"{dataset_name}_token_statistics_sorted.csv")

    # Prepare data: token, frequency, and average norm
    token_data = []
    for token, freq in token_frequency.items():
        cleaned_token = clean_token(token)
        if cleaned_token:  # Only record the token if it's not empty after cleaning
            avg_norm = np.mean(token_norms[token])
            token_data.append((cleaned_token, freq, avg_norm))

    # Sort token data by frequency in descending order
    token_data_sorted = sorted(token_data, key=lambda x: x[1], reverse=True)

    # Save data to CSV
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Token", "Frequency", "Average Norm"])  # Write header
        writer.writerows(token_data_sorted)  # Write sorted data

    print(f"Token statistics saved to {csv_file_path}")

# Function to compute token frequency and aggregate token norms
def compute_token_statistics(tokenizer, input_ids, norms, token_frequency, token_norms):
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().numpy())
    for token, norm in zip(tokens, norms):
        token_frequency[token] += 1
        token_norms[token].append(norm)


# Function to plot the average norm against frequency (log scale)
def plot_norm_vs_frequency(token_frequency, token_norms, output_folder):
    avg_norms = {token: np.mean(norms) for token, norms in token_norms.items()}

    frequencies = np.array([freq for token, freq in token_frequency.items()])
    norms = np.array([avg_norms[token] for token in token_frequency.keys()])


    plt.figure(figsize=(5, 4))

    # Ensure we plot average norms, which is what you want for the y-axis
    plt.scatter(frequencies, norms, alpha=0.6, edgecolor='black')

    plt.ylim(0.2, None)
    # Set x-axis to log scale (as token frequency can vary greatly)
    plt.xscale('log')

    # Labels for x and y axis
    plt.xlabel('Token Frequency (log scale)', fontsize=16)
    plt.ylabel('Average Token Norm', fontsize=16)

    # Set the tick parameters with custom font size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Save or show the plot
    output_path = os.path.join(output_folder, "token_norm_vs_frequency.png")
    plt.savefig(output_path)
    # plt.show()


# Function to plot the average norm against frequency bins (log scale)
def plot_norm_vs_frequency_binned(token_frequency, token_norms, output_folder, bin_count=20):
    # Compute average norm for each token
    avg_norms = {token: np.mean(norms) for token, norms in token_norms.items()}

    # Prepare data for plotting
    frequencies = np.array([freq for token, freq in token_frequency.items()])
    norms = np.array([avg_norms[token] for token in token_frequency.keys()])

    # Define bins for frequencies (log scale bins)
    bins = np.logspace(np.log10(frequencies.min()), np.log10(frequencies.max()), bin_count)

    # Find which bin each frequency falls into
    bin_indices = np.digitize(frequencies, bins)

    # Initialize lists to store average norms and frequencies for each bin
    binned_avg_norms = []
    binned_frequencies = []

    # Compute average norm for each bin
    for i in range(1, len(bins)):
        # Select norms and frequencies in the current bin
        in_bin = (bin_indices == i)
        if np.sum(in_bin) > 0:  # Check if any points fall into this bin
            binned_avg_norms.append(np.mean(norms[in_bin]))
            binned_frequencies.append(np.mean(frequencies[in_bin]))

    # Load custom font (optional, you can remove this if not needed)

    # Create the plot with log scale on x-axis
    plt.figure(figsize=(8, 6))

    # Plot the average norms per bin
    plt.scatter(binned_frequencies, binned_avg_norms, alpha=0.6, edgecolor='black')

    # Set x-axis to log scale
    plt.xscale('log')

    # Labels for x and y axis
    plt.xlabel('Token Frequency (log scale)', fontsize=16)
    plt.ylabel('Average Token Norm (binned)', fontsize=16)

    # Set the tick parameters with custom font size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Save or show the plot
    output_path = os.path.join(output_folder, "binned_token_norm_vs_frequency.png")
    plt.savefig(output_path)
    # plt.show()

def plot_norm_vs_frequency_hist(token_frequency, token_norms, output_folder, bin_count=50):
    # Compute average norm for each token
    avg_norms = {token: np.mean(norms) for token, norms in token_norms.items()}

    # Prepare data for plotting
    frequencies = np.array([freq for token, freq in token_frequency.items()])
    norms = np.array([avg_norms[token] for token in token_frequency.keys()])

    # Define bins for frequencies (log scale bins)
    bins = np.logspace(np.log10(frequencies.min()), np.log10(frequencies.max()), bin_count)

    # Initialize arrays to store the average norm for each bin
    binned_avg_norms = []
    binned_frequencies = []

    # Compute average norm for each bin
    for i in range(1, len(bins)):
        # Find tokens that fall into the current bin
        in_bin = (frequencies >= bins[i-1]) & (frequencies < bins[i])
        if np.sum(in_bin) > 0:  # Only process bins with data
            binned_avg_norms.append(np.mean(norms[in_bin]))
            binned_frequencies.append((bins[i-1] + bins[i]) / 2)  # Use the midpoint of the bin for plotting

    # Adjust the width to match the number of binned frequencies
    bin_widths = np.diff(bins[:len(binned_frequencies) + 1])

    # Create the histogram-like bar plot
    plt.figure(figsize=(8, 6))
    plt.bar(binned_frequencies, binned_avg_norms, width=bin_widths, color='b', alpha=0.7, edgecolor='black')

    # Set x-axis to log scale
    plt.xscale('log')
    plt.ylim(0.2, None)

    # Labels for x and y axis
    plt.xlabel('Token Frequency (log scale)', fontsize=16)
    plt.ylabel('Average Token Norm', fontsize=16)

    # Set the tick parameters
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Save or show the plot
    output_path = os.path.join(output_folder, "binned_token_norm_vs_frequency_hist.png")
    plt.savefig(output_path)
    # plt.show()


def plot_frequency_vs_norm_hist(token_frequency, token_norms, output_folder, bin_count=50, dataset_name=None):
    # Load the custom font
    name_dict = {'gsm8k': 'GSM8K', 'SVAMP': 'SVAMP', 'AQuA': 'AQuA', 'mawps': 'MAWPS'}
    dataset_name = name_dict.get(dataset_name, dataset_name)

    # Load the custom font
    plt.rcParams.update({'font.size': 16})


    # Compute average norm for each token
    avg_norms = {token: np.mean(norms) for token, norms in token_norms.items()}
    frequencies = np.array([freq for token, freq in token_frequency.items()])
    norms = np.array([avg_norms[token] for token in token_frequency.keys()])

    # Define bins for norms
    bins = np.linspace(norms.min(), norms.max(), bin_count)
    binned_avg_frequencies = []
    binned_avg_norms = []

    # Compute average frequency for each norm bin
    for i in range(1, len(bins)):
        in_bin = (norms >= bins[i - 1]) & (norms < bins[i])
        if np.sum(in_bin) > 0:  # Only process bins with data
            binned_avg_frequencies.append(np.mean(frequencies[in_bin]))
            binned_avg_norms.append((bins[i - 1] + bins[i]) / 2)  # Use the midpoint of the bin for plotting

    # Adjust the width to match the number of binned norms
    bin_widths = np.diff(bins[:len(binned_avg_norms) + 1])

    # Normalize norms for color mapping (to create gradient)
    norm_values = np.array(binned_avg_norms)
    norm_min, norm_max = norm_values.min(), norm_values.max()
    norm_normalized = (norm_values - norm_min) / (norm_max - norm_min)  # Normalize to [0, 1]

    # Create colormap based on norm values (using coolwarm colormap)
    colors = cm.coolwarm(norm_normalized)  # Use the coolwarm colormap

    # Create the histogram-like bar plot with transposed axes
    plt.figure(figsize=(5, 4))
    plt.bar(binned_avg_norms, binned_avg_frequencies, width=bin_widths, color=colors, edgecolor='gray')

    # Set y-axis to log scale (for frequency, since token frequencies vary a lot)
    plt.yscale('log')

    # Labels for x and y axis using custom font
    plt.xlabel('Token Norm', fontsize=17)
    plt.ylabel('Token Frequency (log scale)', fontsize=17)
    plt.title(f"{dataset_name}", fontsize=17)

    # Set the tick parameters with custom font size
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)


    # Save or show the plot
    output_path = os.path.join(output_folder, f"{dataset_name}_binned_frequency_vs_norm.pdf")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    output_path = os.path.join(output_folder, f"{dataset_name}_binned_frequency_vs_norm.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset_root', default='dataset', type=str)
    parser.add_argument('--dataset', default='', type=str)
    parser.add_argument('--model', default='LLaMA-7B', type=str)
    parser.add_argument('--adapter', default='LoRA', type=str)
    parser.add_argument('--base_model', required=True)
    # parser.add_argument('--lora_weights', required=True)
    parser.add_argument('--lora_type', default='std')
    parser.add_argument('--lora_r', default=32, type=int)
    parser.add_argument('--load_8bit', action='store_true', default=False)
    # parser.add_argument('--log_wandb', type=int, default=0)
    parser.add_argument('--output_folder', default='figs_last_hidden_state_norm')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    dataset = load_data(args)
    base_model = args.base_model
    load_8bit = args.load_8bit
    ot_pth = f'{args.output_folder}/{args.model}/' 
    create_dir(ot_pth)

    # if args.model == 'LLaMA-7B' or args.model == 'LLMaMA-13B':
    #     tokenizer = LlamaTokenizer.from_pretrained(base_model)
    # else:
    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir = None, trust_remote_code = True)
    if args.model == "gemma-3-4b-it":
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        if args.model in ("gemma-3-4b-it", "gemma-3-1b-it"):
            # model = Gemma3ForCausalLM.from_pretrained(base_model, device_map={"": 0}, attn_implementation='eager', torch_dtype=torch.float16)
            # model.config.use_cache = False
            model = Gemma3ForConditionalGeneration.from_pretrained(base_model, device_map={"": int(os.environ.get("LOCAL_RANK") or 0)}, attn_implementation='eager', torch_dtype=torch.float16)
            model = model.language_model

            model.config.use_cache = False
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                # device_map="auto",
                device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
                trust_remote_code=True,
            )
        # if args.lora_type != 'base':
        #     print('loading lora weights')
        #     model = PeftModel.from_pretrained(
        #         model,
        #         lora_weights,
        #         torch_dtype=torch.float16,
        #         device_map={"":0},
        #         lora_type=args.lora_type,
        #         lora_r=args.lora_r
        #     )

    # Initialize token frequency and norms storage
    token_frequency = defaultdict(int)
    token_norms = defaultdict(list)

    # Compute norms for each instruction in the dataset
    for idx, data in tqdm(enumerate(dataset), total=len(dataset), desc="Computing Token Statistics"):
        instruction = data.get('instruction')
        inputs = tokenizer(instruction, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        input_embeddings = model.get_input_embeddings()(input_ids)

        # output = model(input_ids, output_hidden_states=True)
        # last_hidden_state = output.hidden_states[-1].detach()

        norms = compute_norms(input_embeddings)
        compute_token_statistics(tokenizer, input_ids, norms, token_frequency, token_norms)

    # Plot norm vs frequency
    # plot_norm_vs_frequency(token_frequency, token_norms, args.output_folder)
    # Token frequencies and norms are already computed before calling this function
    # plot_norm_vs_frequency_binned(token_frequency, token_norms, args.output_folder, bin_count=20)
    # Call the function with precomputed token frequencies and norms
    # plot_norm_vs_frequency_hist(token_frequency, token_norms, args.output_folder, bin_count=50)


    plot_frequency_vs_norm_hist(token_frequency, token_norms, ot_pth, bin_count=100, dataset_name=args.dataset)
    save_token_statistics_to_csv(token_frequency, token_norms, ot_pth, args.dataset)
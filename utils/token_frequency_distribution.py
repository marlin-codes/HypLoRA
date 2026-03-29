try:
    import powerlaw  # for power-law fitting
except ImportError:
    powerlaw = None
import torch
import argparse
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer, Gemma3ForConditionalGeneration
import os
import json
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.font_manager import FontProperties
from matplotlib import cm  # For colormap

def mkdirs(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def compute_token_frequencies(dataset, tokenizer, device):
    all_token_ids = []

    # Process each instruction in the dataset and gather token IDs
    for data in tqdm(dataset, desc="Tokenizing"):
        instruction = data.get('instruction')
        inputs = tokenizer(instruction, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        # Convert token ids to a list and append to the global list
        all_token_ids.extend(input_ids.cpu().numpy().flatten().tolist())

    # Count the frequency of each token ID
    token_counter = Counter(all_token_ids)

    # Get the frequency of each unique token ID
    token_frequencies = list(token_counter.values())

    return token_frequencies


def plot_token_frequencies(token_frequencies, output_folder=None, dataset_name=None):
    # Load the custom font
    # name_dict = {'gsm8k': 'GSM8K', 'SVAMP': 'SVAMP', 'AQuA': 'AQuA', 'mawps': 'MAWPS',
    #              'boolq': 'BoolQ', 'winogrande': 'WinoGrande', 'openbookqa': 'OpenBookQA', 'webqsp': 'WebQSP',}
    name_dict = {'arc-c': "ARC-Challenge", 'arc-e': "ARC-Easy"}
    dataset_name = name_dict.get(dataset_name, dataset_name)
    plt.rcParams.update({'font.size': 16})  # Adjust this for global font size

    frequency_counter = Counter(token_frequencies)

    unique_frequencies = sorted(frequency_counter.keys())
    token_counts = [frequency_counter[freq] for freq in unique_frequencies]

    plt.figure(figsize=(5, 4))

    plt.scatter(unique_frequencies, token_counts, color='#5168ce', label='Token Frequency Distribution')

    # Fit the token frequencies using power-law distribution
    if powerlaw is not None:
        fit = powerlaw.Fit(token_frequencies, discrete=True)
        gamma_value = fit.power_law.alpha
        legend_label = f'$\\gamma = {gamma_value:.2f}$'
    else:
        legend_label = 'Token Frequency'

    # Set log scale for both axes
    plt.xscale('log')
    plt.yscale('log')

    # Labels and title with font properties
    plt.xlabel('Token Frequency (log scale)', fontsize=17)
    plt.ylabel('Count (log scale)', fontsize=17)

    # Set the tick parameters with custom font size
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.title(f'{dataset_name}', fontsize=17)

    # Add the legend with the gamma value
    plt.legend([legend_label], prop={'size': 16, 'family': 'sans-serif'})

    # Save or show the plot
    if output_folder:
        mkdirs(output_folder)

        plot_path = os.path.join(output_folder, f"{dataset_name}_token_frequency_distribution.pdf")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plot_path = os.path.join(output_folder, f"{dataset_name}_token_frequency_distribution.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        print('saved plot to', plot_path)
    else:
        plt.show()


def plot_token_frequencies_with_colormap(token_frequencies, output_folder=None):
    # Set global font size for tick labels, axis labels, etc.
    plt.rcParams.update({'font.size': 15})  # Adjust this for global font size

    # Load the custom font

    frequency_counter = Counter(token_frequencies)

    # Unique token frequencies and their counts
    unique_frequencies = sorted(frequency_counter.keys())
    token_counts = np.array([frequency_counter[freq] for freq in unique_frequencies])

    # Log-transform the token frequencies for color mapping (since both axes are in log scale)
    log_frequencies = np.log10(unique_frequencies)

    # Normalize the log-transformed token frequencies for color mapping
    log_freq_min, log_freq_max = log_frequencies.min(), log_frequencies.max()
    normalized_log_frequencies = (log_frequencies - log_freq_min) / (log_freq_max - log_freq_min)  # Normalize to [0, 1]

    # Create colors using the coolwarm colormap based on log-transformed values
    colors = cm.coolwarm(normalized_log_frequencies)

    plt.figure(figsize=(5, 4))

    # Scatter plot with colormap applied
    plt.scatter(unique_frequencies, token_counts, color=colors, label='Token Frequency Distribution', edgecolor='black')

    # Fit the token frequencies using power-law distribution
    if powerlaw is not None:
        fit = powerlaw.Fit(token_frequencies, discrete=True)
        gamma_value = fit.power_law.alpha
        legend_label = f'$\\gamma = {gamma_value:.2f}$'
    else:
        legend_label = 'Token Frequency'

    # Set log scale for both axes
    plt.xscale('log')
    plt.yscale('log')

    # Labels and title with font properties
    plt.xlabel('Tokens\' Frequency (log scale)', fontsize=16)
    plt.ylabel('Counts (log scale)', fontsize=16)

    # Set the tick parameters with larger font size for tick labels
    plt.xticks(fontsize=16)  # Larger font size for x-axis ticks
    plt.yticks(fontsize=16)  # Larger font size for y-axis ticks

    # Add the legend with the gamma value

    # Save or show the plot
    if output_folder:
        # plot_path = os.path.join(output_folder, "token_frequency_distribution_scatter_with_counts.png")
        plot_path = os.path.join(output_folder, "token_frequency_distribution_scatter_with_counts.pdf")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        print('Saved plot to', plot_path)
    else:
        plt.show()



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
    file_path = f'./dataset/{args.dataset}/test.json'
    print(f'evaluate data from {file_path}')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset_root', default='dataset', type=str)
    parser.add_argument('--dataset', default='', type=str)
    parser.add_argument('--model', default='LLaMA-7B', type=str)
    parser.add_argument('--adapter', choices=['LoRA', 'AdapterP', 'AdapterH', 'Parallel', 'Prefix'], required=True)
    parser.add_argument('--base_model', required=True)
    # parser.add_argument('--lora_weights', required=True)
    parser.add_argument('--lora_type', default='std')
    parser.add_argument('--lora_r', default=32, type=int)
    parser.add_argument('--load_8bit', action='store_true', default=False)
    # parser.add_argument('--log_wandb', type=int, default=0)
    parser.add_argument('--output_folder', default='norm_distribution_figs')
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
        tokenizer.padding_side = "right"
    if device == "cuda":
        if args.model == "gemma-3-4b-it":
            model = Gemma3ForConditionalGeneration.from_pretrained(base_model, device_map={"": int(os.environ.get("LOCAL_RANK") or 0)}, attn_implementation='eager')
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

    # Compute token frequencies
    token_frequencies = compute_token_frequencies(dataset, tokenizer, device)

    # Plot the distribution with scatter plot and token counts
    plot_token_frequencies(token_frequencies, ot_pth, args.dataset)
    # plot_token_frequencies_with_colormap(token_frequencies, args.output_folder)
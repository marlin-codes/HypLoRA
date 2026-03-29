
## [Hyperbolic Fine-tuning for LLMs (PDF)](https://arxiv.org/abs/2410.04010)

PyTorch Implementation for [Hyperbolic Fine-tuning for LLMs](https://arxiv.org/abs/2410.04010).


### 1.Introduction
Large language models (LLMs) have demonstrated remarkable performance across various tasks. However, it remains an open question whether the default Euclidean space is the most suitable choice for LLMs. In this study, we investigate the geometric characteristics of LLMs, focusing specifically on tokens and their embeddings. Our findings reveal that token frequency follows a power-law distribution, where high-frequency tokens (e.g., the, that ) constitute the minority, while low-frequency tokens (e.g., apple, dog) constitute the majority. Furthermore, high-frequency tokens cluster near the origin, whereas low-frequency tokens are positioned farther away in the embedding space. Additionally, token embeddings exhibit hyperbolic characteristics, indicating a latent tree-like structure within the embedding space. Motivated by these observations, we propose HypLoRA, an efficient fine-tuning approach that operates in hyperbolic space to exploit these underlying hierarchical structures better. HypLoRA performs low-rank adaptation directly in hyperbolic space, thereby preserving hyperbolic modeling capabilities throughout the fine-tuning process. Extensive experiments across various base models and reasoning benchmarks, specifically arithmetic and commonsense reasoning tasks, demonstrate that HypLoRA substantially improves LLM performance.

### 2.Power-law Distribution in Token Embedding

Check the code at [utils/token_frequency_distribution.py](utils/token_frequency_distribution.py)

| ![GSM8K Token Frequency](./utils/results/figs_frequency/gsm8k/GSM8K_token_frequency_distribution.png)  | ![AQuA Token Frequency](./utils/results/figs_frequency/AQuA/AQuA_token_frequency_distribution.png)  | ![BoolQ Token Frequency](./utils/results/figs_frequency/boolq/BoolQ_token_frequency_distribution.png)  |
|:----------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------:|
| ![Math 10K Token Frequency](./utils/results/figs_frequency/math_10k/math_10k_token_frequency_distribution.png) | ![Math 50K Token Frequency](./utils/results/figs_frequency/math_50k/math_50k_token_frequency_distribution.png) | ![MAWPS Token Frequency](./utils/results/figs_frequency/mawps/MAWPS_token_frequency_distribution.png) |
| ![OpenBookQA Token Frequency](./utils/results/figs_frequency/openbookqa/OpenBookQA_token_frequency_distribution.png) | ![SVAMP Token Frequency](./utils/results/figs_frequency/SVAMP/SVAMP_token_frequency_distribution.png) | ![WinoGrande Token Frequency](./utils/results/figs_frequency/winogrande/WinoGrande_token_frequency_distribution.png) |

### 3. Hierarchical examples in Token Embedding

<div align="center">
    <img src="./figs/number_hierarchy.png" alt="img.png">
</div>


![img.png](./figs/numbers.png)

### 4. Frequency Distribution w.r.t. Norm

Check the code at [utils/token_norm_plot.py](./utils/token_norm_plot.py)

| ![AQuA Frequency vs Norm](./utils/results/figs_frequency_norm/AQuA/AQuA_binned_frequency_vs_norm.png)  | ![BoolQ Frequency vs Norm](./utils/results/figs_frequency_norm/boolq/boolq_binned_frequency_vs_norm.png)  | ![GSM8K Frequency vs Norm](./utils/results/figs_frequency_norm/gsm8k/GSM8K_binned_frequency_vs_norm.png)  |
|:-----------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------:|
| ![Math 10K Frequency vs Norm](./utils/results/figs_frequency_norm/math_10k/math_10k_binned_frequency_vs_norm.png) | ![Math 50K Frequency vs Norm](./utils/results/figs_frequency_norm/math_50k/math_50k_binned_frequency_vs_norm.png) | ![MAWPS Frequency vs Norm](./utils/results/figs_frequency_norm/mawps/MAWPS_binned_frequency_vs_norm.png) |
| ![OpenBookQA Frequency vs Norm](./utils/results/figs_frequency_norm/openbookqa/openbookqa_binned_frequency_vs_norm.png) | ![SVAMP Frequency vs Norm](./utils/results/figs_frequency_norm/SVAMP/SVAMP_binned_frequency_vs_norm.png) | ![WinoGrande Frequency vs Norm](./utils/results/figs_frequency_norm/winogrande/winogrande_binned_frequency_vs_norm.png) |



### 5. [Core Steps for Hyperbolic fine-tuning LLMs](https://github.com/marlin-codes/HypLoRA/blob/main/peft/tuners/lora/layer.py#L822)

```
(0) Curvature parameterization
(1) map: Euclidean -> Hyperboloid
(2) Low-rank adaptation on the manifold
(3) Lorentz constraint: re-project onto hyperboloid
(4) map: Hyperboloid -> Euclidean
```


### 6. Setup

```bash
# Install dependencies
bash setup.sh

# Activate environment
conda activate adapter
export PYTHONPATH=/path/to/HypLoRA/peft:$PYTHONPATH
export HF_HOME=<your_huggingface_cache>
export WANDB_DISABLED=true
```

### 7. Running Scripts

Each model has its own `finetune.sh` and `eval.sh` under `scripts/`.

#### Table 3 (Math: AQuA, mawps, SVAMP, gsm8k)

```bash
# Example: train and evaluate qwen with hyplora
bash scripts/table3/hyplora/qwen2.5-7b/finetune.sh
bash scripts/table3/hyplora/qwen2.5-7b/eval.sh

# Available models: qwen2.5-7b, llama3-8b, gemma3-4b, gemma-7b
# Available methods: hyplora, hyplora_simplified, lora
```

#### Table 4 (Commonsense: boolq, piqa, hellaswag, winogrande, ARC, openbookqa, social_iqa)

```bash
# Example: train and evaluate qwen with hyplora
bash scripts/table4/hyplora/qwen2.5-7b/finetune.sh
bash scripts/table4/hyplora/qwen2.5-7b/eval.sh
```

### 8. Evaluation Batch Sizes

Scripts default to batch_size=1 for maximum compatibility. To speed up evaluation on high-memory GPUs (e.g. H100 80GB), increase the batch size in the scripts:

#### Table 3 (Math)

| Dataset | Recommended |
|---------|-------------|
| AQuA | 5 |
| mawps | 10 |
| SVAMP | 10 |
| gsm8k | 5 |

#### Table 4 (Commonsense)

| Dataset | qwen/llama/gemma3 | gemma7 |
|---------|-------------------|--------|
| boolq, social_i_qa, winogrande, openbookqa | 100 | 50 |
| piqa, ARC-Easy, ARC-Challenge | 40 | 30 |
| hellaswag | 40 | 20 |

### BibTex
```
@inproceedings{yang2025hyperbolicfinetuning,
  title     = {Hyperbolic Fine-Tuning for Large Language Models},
  author    = {Menglin Yang and Ram Samarth B B and Aosong Feng and Bo Xiong and Jihong Liu and Irwin King and Rex Ying},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {38},
  year      = {2025},
  url       = {https://arxiv.org/abs/2410.04010}
}
```
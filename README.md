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

Clone the HypLoRA repository and switch to the checkpoint release branch:

```bash
git clone --branch hf-checkpoint-v https://github.com/marlin-codes/HypLoRA.git
cd HypLoRA
```

Create the Conda environment and install the required dependencies:

```bash
bash setup.sh
conda activate hyplora
```

The setup script installs the pinned dependencies, including PyTorch, FlashAttention-2, Transformers, and PEFT. It then patches the installed PEFT package with the customized HypLoRA implementation and verifies that the correct files are being imported.

The environment name is `hyplora`. Activate it before running training or evaluation scripts.

### 7. Running Training and Evaluation Scripts

We provide `finetune.sh` and `eval.sh` scripts for **HypLoRA** and **LoRA** on the following model families:

- Gemma-7B
- Llama 3-8B
- Gemma 3-4B
- Qwen 2.5-7B

The scripts are organized according to the training dataset:

| Script directory | Dataset | Purpose |
|---|---|---|
| `scripts/table3/` | Math-10K | Mathematical reasoning |
| `scripts/table4/` | Commonsense-170K | Commonsense reasoning |

Each method and model has its own directory containing `finetune.sh` and `eval.sh`.

#### Table 3: Math-10K

For example, to fine-tune HypLoRA on Math-10K using Llama 3-8B, navigate to the model-specific script directory:

```bash
cd scripts/table3/hyplora/llama3-8b/
```

Run fine-tuning:

```bash
bash finetune.sh
```

After fine-tuning completes, run evaluation using the checkpoint configured in the corresponding `eval.sh`:

```bash
bash eval.sh
```

For standard LoRA, use the corresponding `lora` directory instead:

```bash
cd scripts/table3/lora/llama3-8b/
bash finetune.sh
bash eval.sh
```

The same structure applies to Gemma-7B, Gemma 3-4B and Qwen 2.5-7B.

The Table 3 evaluation scripts cover AQuA, MAWPS, SVAMP, and GSM8K.

#### Table 4: Commonsense-170K

To fine-tune HypLoRA on Commonsense-170K using Llama 3-8B:

```bash
cd scripts/table4/hyplora/llama3-8b/
bash finetune.sh
```

Evaluate the fine-tuned checkpoint:

```bash
bash eval.sh
```

For standard LoRA, use:

```bash
cd scripts/table4/lora/llama3-8b/
bash finetune.sh
bash eval.sh
```

The same directory structure applies to Gemma 3-4B and Qwen 2.5-7B.

The Table 4 evaluation scripts cover BoolQ, PIQA, SocialIQA, HellaSwag, WinoGrande, ARC-Challenge, ARC-Easy, and OpenBookQA.

**Note:** The shell scripts determine the project root relative to their own locations. Run the commands from the corresponding model-specific directory as shown above. Check the configured output and checkpoint paths if you change the default training settings.

### 8. Using Released HypLoRA Checkpoints

We provide pretrained HypLoRA adapter checkpoints for Math-10K and Commonsense-170K. These are **PEFT adapter weights, not complete language models**. The corresponding base model and the HypLoRA implementation from this repository are required.

#### Table 3: Math-10K Checkpoints

| Base model | Hugging Face checkpoint |
|---|---|
| Gemma-7B   | [GGLabYale/hyplora-gemma-7b-math10k](https://huggingface.co/GGLabYale/hyplora-gemma-7b-math10k)
| Llama 3-8B | [GGLabYale/hyplora-llama3-8b-math10k](https://huggingface.co/GGLabYale/hyplora-llama3-8b-math10k) |
| Gemma 3-4B | [GGLabYale/hyplora-gemma3-4b-math10k](https://huggingface.co/GGLabYale/hyplora-gemma3-4b-math10k) |
| Qwen 2.5-7B | [GGLabYale/hyplora-qwen25-7b-math10k](https://huggingface.co/GGLabYale/hyplora-qwen25-7b-math10k) |

#### Table 4: Commonsense-170K Checkpoints

| Base model | Hugging Face checkpoint |
|---|---|
| Llama 3-8B | [GGLabYale/hyplora-llama3-8b-commonsense170k](https://huggingface.co/GGLabYale/hyplora-llama3-8b-commonsense170k) |
| Gemma 3-4B | [GGLabYale/hyplora-gemma3-4b-commonsense170k](https://huggingface.co/GGLabYale/hyplora-gemma3-4b-commonsense170k) |
| Qwen 2.5-7B | [GGLabYale/hyplora-qwen25-7b-commonsense170k](https://huggingface.co/GGLabYale/hyplora-qwen25-7b-commonsense170k) |

#### Evaluating with a Released Checkpoint

To evaluate using one of our released adapters instead of a locally fine-tuned checkpoint:

1. Navigate to the appropriate model-specific `eval.sh` directory.
2. Locate the `--lora_weights` argument.
3. Replace its local checkpoint path with the corresponding Hugging Face repository ID from the tables above.
4. Run `bash eval.sh` from that directory.

For example, to evaluate using the released Llama 3-8B Math-10K HypLoRA checkpoint, edit `scripts/table3/hyplora/llama3-8b/eval.sh` to use:

```bash
--lora_weights "GGLabYale/hyplora-llama3-8b-math10k"
```

Then run:

```bash
cd scripts/table3/hyplora/llama3-8b/
bash eval.sh
```

For a locally fine-tuned checkpoint, keep `--lora_weights` pointed at the corresponding local checkpoint or output directory.


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
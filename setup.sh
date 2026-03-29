#!/bin/bash
# =============================================================================
# HyperLoRA Setup Script
# Run this once after cloning the repo to set up the environment.
# =============================================================================

set -euo pipefail

echo "[1/4] Creating conda environment..."
conda create -n hyperlora python=3.10 -y 2>/dev/null || echo "Environment 'hyperlora' already exists, skipping."
eval "$(conda shell.bash hook)"
conda activate hyperlora

echo "[2/4] Installing PyTorch (CUDA 12.1)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q

echo "[3/4] Installing dependencies..."
pip install -r requirements.txt -q

echo "[4/4] Installing custom peft with HyperLoRA extensions..."
# Install official peft first, then patch with our modified layer.py
pip install peft==0.15.2 -q
# Find installed peft location and patch
PEFT_DIR=$(python -c "import peft, os; print(os.path.dirname(peft.__file__))")
echo "Patching ${PEFT_DIR}/tuners/lora/layer.py with HyperLoRA extensions..."
cp peft/src/peft/tuners/lora/layer.py "${PEFT_DIR}/tuners/lora/layer.py"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Before running experiments, set these environment variables:"
echo ""
echo "  export HF_HOME=<path_to_huggingface_cache>"
echo "  export TRANSFORMERS_CACHE=<path_to_transformers_cache>"
echo "  export WANDB_DISABLED=true"
echo ""
echo "Then run:"
echo "  conda activate hyperlora"
echo "  bash scripts/final_test/table3/gpu0_qwen2.5-7b.sh"
echo ""

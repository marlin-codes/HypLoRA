#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

cd "$PROJECT_ROOT" || exit 1
unset PYTHONPATH

command -v "$PYTHON" >/dev/null 2>&1 || {
    echo "ERROR: Python executable not found: $PYTHON"
    echo "Activate the HypLoRA environment first."
    exit 1
}

echo "Project root: $PROJECT_ROOT"
echo "Python: $("$PYTHON" -c 'import sys; print(sys.executable)')"

"$PYTHON" -P - <<'PY'
import importlib
from pathlib import Path
import peft

peft_root = Path(peft.__file__).resolve().parent

if "site-packages" not in peft_root.parts:
    raise SystemExit(
        f"ERROR: Expected installed PEFT, got: {peft_root}"
    )

print("PEFT:", peft.__file__)

for name in (
    "peft.tuners.lora.layer",
    "peft.tuners.lora.model",
    "peft.tuners.lora.config",
):
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()

    if not path.is_relative_to(peft_root):
        raise SystemExit(f"ERROR: Unexpected module path: {path}")

    print(f"Verified: {name} -> {path}")
PY

"$PYTHON" -P -u "$PROJECT_ROOT/finetune_gemma7b.py" \
    --base_model "google/gemma-7b" \
    --data_path './ft-training_set/math_10k.json' \
    --output_dir './trained_models/gemma-7b-lora-r32-math' \
    --batch_size 16 --micro_batch_size 4 \
    --num_epochs 3 --learning_rate 3e-4 \
    --cutoff_len 256 --val_set_size 120 \
    --lora_r 32 --lora_alpha 64 \
    --target_modules '["q_proj", "v_proj", "k_proj", "up_proj", "down_proj"]' \
    --lora_type "std"
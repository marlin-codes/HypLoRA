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

"$PYTHON" -P -u "$PROJECT_ROOT/evaluate.py" \
    --dataset 'AQuA' --model 'gemma-7b' \
    --base_model 'google/gemma-7b' \
    --lora_weights './trained_models/gemma-7b-lora-r32-math' \
    --lora_type 'std' --lora_alpha 64 \
    --rank 32 --adapter 'LoRA' --run 0 --batch_size 5
"$PYTHON" -P -u "$PROJECT_ROOT/evaluate.py" \
    --dataset 'mawps' --model 'gemma-7b' \
    --base_model 'google/gemma-7b' \
    --lora_weights './trained_models/gemma-7b-lora-r32-math' \
    --lora_type 'std' --lora_alpha 64 \
    --rank 32 --adapter 'LoRA' --run 0 --batch_size 10
"$PYTHON" -P -u "$PROJECT_ROOT/evaluate.py" \
    --dataset 'SVAMP' --model 'gemma-7b' \
    --base_model 'google/gemma-7b' \
    --lora_weights './trained_models/gemma-7b-lora-r32-math' \
    --lora_type 'std' --lora_alpha 64 \
    --rank 32 --adapter 'LoRA' --run 0 --batch_size 10
"$PYTHON" -P -u "$PROJECT_ROOT/evaluate.py" \
    --dataset 'gsm8k' --model 'gemma-7b' \
    --base_model 'google/gemma-7b' \
    --lora_weights './trained_models/gemma-7b-lora-r32-math' \
    --lora_type 'std' --lora_alpha 64 \
    --rank 32 --adapter 'LoRA' --run 0 --batch_size 5
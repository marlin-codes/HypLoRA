# Third-party licenses and scope

The [HypLoRA Research Use License](LICENSE) covers only original project material
in its explicit path list. Its commercial-use permission requirement does not
apply to third-party code or rights already granted under other licenses. This
index preserves identified sources and notices; it is not a claim that every
file or portion without a header is original.

## PEFT and existing permissive grants

All of `peft/` is excluded from the research-use license. This includes HypLoRA
changes already published inside files carrying permissive headers. The
[initial public revision](https://github.com/marlin-codes/HypLoRA/commit/66df99a412c6f1166073127f3f9da535dbc084c0)
contains those headers and the existing HypLoRA implementation. No earlier valid
grant is withdrawn or made conditional on buying a commercial license.

- 135 of the 137 baseline files in `peft/` carry Apache-2.0 headers attributed to
  the HuggingFace Inc. team (2023-, 2024- or 2025-present). The complete
  [Apache-2.0 text](LICENSES/Apache-2.0.txt) is included. The file headers remain
  unchanged. Upstream: [Hugging Face PEFT](https://github.com/huggingface/peft).
- `peft/tuners/lora/layer.py` additionally credits Microsoft LoRA, including
  `Copyright (c) Microsoft Corporation. All rights reserved.` and an MIT notice.
  `peft/utils/save_and_load.py` also credits LoRA utilities. Preserve those
  notices with the [Microsoft LoRA MIT text](LICENSES/MIT-LoRA.txt), copied from
  the [upstream revision referenced in the source](https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/LICENSE.md).
- `peft/tuners/_buffer_dict.py` has a Meta Platforms copyright and MIT header,
  crediting BoTorch. The [BoTorch MIT text](LICENSES/MIT-BoTorch.txt) preserves
  `Copyright (c) Meta Platforms, Inc. and affiliates.` Source of the text:
  [BoTorch](https://github.com/pytorch/botorch/blob/b0dd01be61d517c387e234903a6e2cc38c057ea2/LICENSE).
  The precise original copied BoTorch revision is not recorded in HypLoRA.
- `peft/tuners/boft/fbd/__init__.py` is empty and has no header. Its inclusion in
  `peft/` does not subject it to the research-use restriction.

The vendored version string `0.15.2.dev0` and installation pin `0.15.2` do not
establish an exact copied upstream snapshot. Comparison with
[PEFT v0.15.2](https://github.com/huggingface/peft/tree/3c7b6e7f0252cb18386d86b056a9b100a8160792/src/peft)
found 75 identical files, 54 different files and 8 paths absent from that tag.
Differences are not proof of original project authorship. The tree also retains
embedded source references, including Transformers, Accelerate, Diffusers,
LyCORIS, scikit-learn, CorDA, LoftQ, NeMo, P-tuning-v2, LLaMA-Adapter, Microsoft
MTTL, LoRA-Torch and kohya/sd-scripts. Their applicable rights and notice duties
are preserved; this index does not certify an exhaustive upstream audit.

## MIT-derived delta utilities

The complete `delta_hyp` and `get_delta` functions in each of
[`utils/delta_hyperbolicity.py`](utils/delta_hyperbolicity.py) and
[`utils/run_delta_all.py`](utils/run_delta_all.py) are excluded from the
research-use restriction and retain the [MIT license](LICENSES/MIT-Hyperbolic-Image-Embeddings.txt).
They derive from
[`hyptorch/delta.py`](https://github.com/leymir/hyperbolic-image-embeddings/blob/6633edbbeffd6d90271f0963852a046c64f407d6/hyptorch/delta.py)
in Valentin Khrulkov's Hyperbolic Image Embeddings repository.

**Copyright (c) 2019 Valentin Khrulkov.** The complete upstream
[MIT text](https://github.com/leymir/hyperbolic-image-embeddings/blob/6633edbbeffd6d90271f0963852a046c64f407d6/LICENSE)
is included without modification. The executable body of `delta_hyp` matches the
upstream function. `get_delta` is adapted to accept token-embedding tensors and
configurable sampling instead of extracting image features through VGG. Source
comments mark the beginning and end of these two-function blocks. The remaining
original project orchestration and plotting code is covered by the root license.
The MIT permission for the marked functions includes commercial use under MIT's
own conditions and does not require separate permission from marlin-codes.

## Mixed-origin training and evaluation scripts

The root `finetune_gemma.py`, `finetune_gemma7b.py`, `finetune_llama3.py`,
`finetune_qwen.py`, `evaluate.py`, `commonsense_evaluate.py`,
`commonsense_evaluate_gemma.py` and `commonsense_evaluate_qwen.py` must not be
treated as wholly original merely because they lack license headers. Original
project contributions are covered by the root license; copied/adapted upstream
portions remain excluded and retain applicable upstream permissions.

The baseline `finetune_qwen.py` lines 180–212 match 33 contiguous lines of both
[Alpaca-LoRA's `finetune.py`](https://github.com/tloen/alpaca-lora/blob/8bb8579e403dc78e37fe81ffbb253c413007323f/finetune.py#L212)
and [LLM-Adapters' `finetune.py`](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/816657208af4db747803f87ba40a4c71383fed7a/finetune.py#L253).
Their respective [Alpaca-LoRA license](https://github.com/tloen/alpaca-lora/blob/8bb8579e403dc78e37fe81ffbb253c413007323f/LICENSE)
and [LLM-Adapters license](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/816657208af4db747803f87ba40a4c71383fed7a/LICENSE)
are Apache-2.0; its full text is included in `LICENSES/Apache-2.0.txt`.
These matches are positive provenance evidence, not an exhaustive boundary of
all shared code or proof of the exact import history. The root license's
third-party and prior-grant exclusions apply even where a copied portion is not
individually annotated. Preserve any additional notices from the actual source
distribution when reusing such material.

## Data, figures, models and outputs

All JSON files in `dataset/` and `ft-training_set/`, all material in `figs/`,
`utils/figs/` and `utils/results/`, and referenced/downloaded models, tokenizers,
weights and checkpoints are outside the root code license. No blanket permission
or new research-use restriction is placed on them. Their source-specific terms
and permissions must be assessed separately. The original code in
`dataset/mawps/data_process.py` is listed in the root license, but its input and
output data are not licensed by that code license.

Independent generated outputs are not automatically covered by the code license.
Any protected code or third-party content they reproduce keeps its applicable
terms. Dependencies also retain their own licenses; listing a dependency in
`requirements.txt` does not relicense it.

The file counts and comparison evidence above refer to baseline
`4e919482a8bda623a12d8251957a78547f42b919`. They do not claim that a future version
has identical provenance. The exclusions in `LICENSE` control regardless.

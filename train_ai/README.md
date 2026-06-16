# Seeds of Truth canonical LoRA trainer

A single, generic LoRA fine-tuning pipeline distilled from the
project's iterative training notebooks. Two training modes are
supported through one configuration: continued pretraining on a raw
text corpus (`mode="corpus"`) and instruction tuning on Q/A pairs
(`mode="qa"`). The pipeline prefers Unsloth's fast 4-bit / fused-kernel
path when it's installed, and falls back to the vanilla HuggingFace
stack (`transformers` + `peft` + `trl`) otherwise.

## Install

```
pip install -r seedsoftruth/train_ai/requirements.txt
```

For the Unsloth fast path:

```
pip install -r seedsoftruth/train_ai/requirements-unsloth.txt
```

Unsloth has its own GPU/CUDA constraints — see the
[Unsloth README](https://github.com/unslothai/unsloth) for the supported
matrix. The trainer works on plain `transformers` if Unsloth isn't
available; you'll just be slower and use more VRAM.

## What's in here

| File                  | Purpose                                                       |
| --------------------- | ------------------------------------------------------------- |
| `train.py`            | Core trainer: `TrainingConfig`, `train()`, CLI.               |
| `train.ipynb`         | Thin notebook wrapper that walks through the same pipeline.   |
| `merge_lora.py`       | Fuse a trained adapter back into its base model.              |
| `gen_qa_pairs.py`     | Generate QA training pairs from a raw text corpus via an LLM. |
| `tokenizer_tools.py`  | Stop-token utilities and GGUF→HF conversion.                  |

## Two training modes

### `corpus` — continued pretraining on raw text

Reads every `.txt` file in `--dataset-dir`, concatenates with EOS
between documents, packs into fixed-size blocks, and trains the LoRA
adapter on next-token prediction. Use this when you want the base
model to absorb the *style and content* of a domain corpus.

```
python -m seedsoftruth.train_ai \
    --mode corpus \
    --base-model unsloth/Qwen3-8B \
    --dataset-dir ./corpus/text \
    --output-dir ./lora_out \
    --epochs 1 --lora-rank 16
```

### `qa` — instruction tuning on Q/A pairs

Reads `.json` and `.jsonl` files containing
`{"question": "...", "answer": "..."}` records and renders each pair as
`"Q: {q}\nA: {a}</s>"` before tokenization. Both JSON arrays and JSONL
are accepted; format is sniffed per file. Use this when you have a
labeled QA dataset and want the model to learn the Q/A response shape.

```
python -m seedsoftruth.train_ai \
    --mode qa \
    --base-model unsloth/Qwen3-8B \
    --dataset-dir ./qa_pairs \
    --output-dir ./lora_out \
    --epochs 1 --lora-rank 16
```

## Python API

```python
from seedsoftruth.train_ai import TrainingConfig, train

cfg = TrainingConfig(
    base_model="unsloth/Qwen3-8B",
    dataset_dir="./corpus/text",
    output_dir="./lora_out",
    mode="corpus",
    lora_rank=16,
    num_train_epochs=1,
)
trained = train(cfg)
```

The trainer returns the wrapped PEFT model so you can keep working
with it programmatically — for example, to merge immediately after
training.

## Key configuration knobs

| Field                            | Default           | What it controls                                              |
| -------------------------------- | ----------------- | ------------------------------------------------------------- |
| `base_model`                     | (required)        | HF repo or local path of the base model.                      |
| `dataset_dir`                    | (required)        | Directory of `.txt` (corpus) or `.json/.jsonl` (qa).          |
| `output_dir`                     | (required)        | Where checkpoints + final adapter are written.                |
| `mode`                           | `"corpus"`        | `"corpus"` or `"qa"`.                                         |
| `block_size`                     | `1024`            | Tokens per training block.                                    |
| `eval_split_frac`                | `0.1`             | Held-out fraction; `0` disables eval and early stopping.      |
| `lora_rank` / `lora_alpha`       | `16` / `32`       | LoRA matrix shape.                                            |
| `lora_target_modules`            | `q,k,v,o`         | Which projections to wrap. Most LLaMA/Qwen variants use these.|
| `num_train_epochs`               | `1`               |                                                               |
| `per_device_train_batch_size`    | `8`               | On a 24 GB GPU, lower this for larger models.                 |
| `gradient_accumulation_steps`    | `2`               | Effective batch size = `batch_size * grad_accum * world_size`.|
| `learning_rate`                  | `1e-5`            | Start here; halve for >32B models, double for <2B.            |
| `dtype`                          | `"bfloat16"`      | `bfloat16`, `float16`, `auto`.                                |
| `load_in_4bit`                   | `True`            | Disable for full-precision training (large VRAM hit).         |
| `resume_from_checkpoint`         | `None`            | Path to a `checkpoint-N/` directory.                          |
| `push_to_hub_repo`               | `None`            | If set, upload the adapter to this HF repo when done.         |

The CLI exposes the same fields as kebab-case flags. Run
`python -m seedsoftruth.train_ai --help` for the full list.

## End-to-end workflow

```bash
# 1. (Optional) Generate QA pairs from your corpus.
python -m seedsoftruth.train_ai.gen_qa_pairs \
    --model unsloth/Qwen2.5-7B-Instruct \
    --corpus-dir ./corpus/text \
    --output-dir ./qa_pairs

# 2. Train a LoRA adapter.
python -m seedsoftruth.train_ai \
    --mode qa \
    --base-model unsloth/Qwen3-8B \
    --dataset-dir ./qa_pairs \
    --output-dir ./lora_out \
    --epochs 1 --lora-rank 16 \
    --push-to-hub-repo your-org/your-lora-v1

# 3. Merge the adapter into the base model.
python -m seedsoftruth.train_ai.merge_lora \
    --base-model unsloth/Qwen3-8B \
    --adapter ./lora_out \
    --output-dir ./merged_out \
    --push-to-hub your-org/your-merged-v1
```

## Notebook companion

`train.ipynb` walks through the same pipeline cell-by-cell, importing
the module functions instead of redefining them. Edit the
`TrainingConfig` cell, then run the notebook top-to-bottom — every
subsequent cell calls into `train.py`.

## Known constraints

- **GPU.** This is LoRA training of LLMs. You need a CUDA GPU with
  enough VRAM to hold the base model in 4-bit. As a rough guide: 8 GB
  for 7B-class models, 24 GB for 13B-class, 80 GB for 70B-class.
- **Unsloth Linux-only.** Unsloth currently only supports Linux. On
  macOS or Windows, omit it and the trainer falls back to plain
  `transformers`.
- **`load_in_4bit` + `gradient_checkpointing`** is the default. If you
  see OOM on small models, those are the first knobs to try keeping;
  if you're seeing slowness on large GPUs, set `load_in_4bit=False`
  and lower `gradient_accumulation_steps`.
- **Push-to-hub** requires `HF_TOKEN` in the environment.

## Provenance

This pipeline consolidates the project's training notebook lineage:

- `train_lora_from_corpus*` → corpus mode
- `train_lora_health_*_qa*` → qa mode
- `train_gamma_llama3_70b_v9` and `train_beta_slim_*` → LoRA hyperparameters
- `merge_lora_with_base_model_*` → `merge_lora.py`
- `gen_qa_pairs_beta_plus*` and `_overlaps2` → `gen_qa_pairs.py`
- `add_stop_token`, `replace_stop_token`, `convert_gguf_to_hf*` → `tokenizer_tools.py`

Hyperparameter defaults (LR, batch size, LoRA rank/alpha, etc.) are
the median of those notebooks. Project-specific paths, model repo
names, and dataset names have been removed.

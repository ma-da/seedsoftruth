"""
Canonical LoRA trainer for Seeds of Truth.

Two training modes are supported through one pipeline:

    corpus  — continued pretraining on raw .txt files. The corpus is
              concatenated with EOS between documents and chunked into
              fixed-size blocks. No instruction structure is imposed.

    qa      — instruction tuning on Q/A pairs loaded from JSON or JSONL
              files with `{"question": ..., "answer": ...}` records.
              Each pair is rendered as ``"Q: {q}\\nA: {a}</s>"`` before
              tokenization.

The trainer prefers the Unsloth fast path
(``unsloth.FastLanguageModel``) when it is installed, and falls back
to the vanilla HuggingFace stack (``transformers`` +
``peft.get_peft_model`` + ``trl.SFTTrainer``) when it is not.

Public surface
--------------

    TrainingConfig            knobs the trainer reads.
    load_model_and_tokenizer  open the base model. Unsloth-first.
    apply_lora                wrap the model with a LoRA adapter.
    load_corpus_dataset       read .txt files into a flat list of strings.
    load_qa_dataset           read .json/.jsonl QA pairs into Q:/A: strings.
    tokenize_and_pack         append-EOS tokenize + pack into fixed blocks.
    build_trainer             assemble TrainingArguments + Trainer.
    train                     end-to-end driver: returns the trained model.
    push_to_hub               upload the LoRA adapter to a HF repo.
    main                      CLI entry point.

Run from the command line:

    python -m seedsoftruth.train_ai \\
        --mode corpus \\
        --base-model unsloth/Qwen3-8B \\
        --corpus-dir ./corpus/text \\
        --output-dir ./lora_out \\
        --epochs 1 --lora-rank 16

This module is the rewritten descendant of the project's
``train_lora_*`` notebook lineage (gamma-llama3-70b, from-corpus-qwen3,
health-tiny-qa, beta-slim). The hyperparameter defaults are the
median of those notebooks; project-specific paths and repo names have
been removed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

log = logging.getLogger(__name__)

TrainingMode = Literal["corpus", "qa"]


# --------------------------------------------------------------------------- #
# config                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class TrainingConfig:
    """Every knob the trainer reads.

    The required fields are ``base_model``, ``dataset_dir``, and
    ``output_dir``. Everything else is a sensible default.
    """

    # ---- required ---- #
    base_model: str
    dataset_dir: str
    output_dir: str

    # ---- mode + dataset shape ---- #
    mode: TrainingMode = "corpus"
    block_size: int = 1024
    eval_split_frac: float = 0.1   # set to 0.0 to disable eval

    # ---- model load ---- #
    max_seq_length: int = 1024
    load_in_4bit: bool = True
    trust_remote_code: bool = True
    dtype: str = "bfloat16"        # "bfloat16" | "float16" | "auto"

    # ---- LoRA ---- #
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    lora_use_rslora: bool = False
    lora_use_gradient_checkpointing: bool = True

    # ---- training arguments ---- #
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.02
    max_grad_norm: float = 0.3
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    save_total_limit: int = 2
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    report_to: str = "none"
    dataloader_num_workers: int = 4
    seed: int = 42

    # ---- runtime control ---- #
    resume_from_checkpoint: Optional[str] = None
    push_to_hub_repo: Optional[str] = None
    push_to_hub_private: bool = True

    def __post_init__(self) -> None:
        if self.mode not in ("corpus", "qa"):
            raise ValueError(f"mode must be 'corpus' or 'qa', got {self.mode!r}")
        if not (0.0 <= self.eval_split_frac < 1.0):
            raise ValueError("eval_split_frac must be in [0, 1)")


# --------------------------------------------------------------------------- #
# model loading                                                               #
# --------------------------------------------------------------------------- #

def _resolve_dtype(name: str):
    import torch
    return {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(name, "auto")


def load_model_and_tokenizer(config: TrainingConfig) -> Tuple[object, object]:
    """Load the base model and tokenizer. Tries Unsloth first.

    Returns (model, tokenizer). The model is the unwrapped base model;
    call ``apply_lora`` next.
    """
    try:
        from unsloth import FastLanguageModel  # type: ignore

        log.info("loading base model via Unsloth: %s", config.base_model)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.base_model,
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit,
            dtype=_resolve_dtype(config.dtype),
            trust_remote_code=config.trust_remote_code,
            device_map="auto",
        )
        return model, tokenizer

    except ImportError:
        log.warning("Unsloth not installed; falling back to vanilla HF stack")

    # Vanilla HF fallback. Works for any HF causal LM.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("loading base model via transformers: %s", config.base_model)

    load_kwargs = dict(
        trust_remote_code=config.trust_remote_code,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    dtype_resolved = _resolve_dtype(config.dtype)
    if config.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=(
                    torch.bfloat16 if dtype_resolved == "auto" else dtype_resolved
                ),
            )
        except ImportError:
            log.warning("bitsandbytes not installed; loading without 4-bit quantization")
            if dtype_resolved != "auto":
                load_kwargs["torch_dtype"] = dtype_resolved
    else:
        if dtype_resolved != "auto":
            load_kwargs["torch_dtype"] = dtype_resolved

    model = AutoModelForCausalLM.from_pretrained(config.base_model, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        use_fast=True,
        trust_remote_code=config.trust_remote_code,
    )
    return model, tokenizer


# --------------------------------------------------------------------------- #
# LoRA                                                                        #
# --------------------------------------------------------------------------- #

def apply_lora(model, tokenizer, config: TrainingConfig):
    """Wrap `model` with a LoRA adapter. Reuses Unsloth's helper if loaded
    that way, otherwise uses `peft.get_peft_model`. Returns the wrapped model.

    Also harmonizes pad_token / pad_token_id between the tokenizer and model
    config so SFTTrainer doesn't trip on missing pad tokens.
    """
    # Pad token harmonization (run before LoRA wrap).
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        log.info("set tokenizer.pad_token to eos_token")

    # Prefer Unsloth's helper when the model came from Unsloth.
    try:
        from unsloth import FastLanguageModel  # type: ignore

        peft_model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            use_gradient_checkpointing=config.lora_use_gradient_checkpointing,
            modules_to_save=None,
            use_rslora=config.lora_use_rslora,
        )
    except ImportError:
        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(model, lora_cfg)
        if config.lora_use_gradient_checkpointing and hasattr(
            peft_model, "gradient_checkpointing_enable"
        ):
            peft_model.gradient_checkpointing_enable()

    # Mirror token IDs onto both configs.
    for cfg in (model.config, peft_model.config):
        cfg.bos_token_id = tokenizer.bos_token_id
        cfg.eos_token_id = tokenizer.eos_token_id
        cfg.pad_token_id = tokenizer.pad_token_id
    peft_model.config.use_cache = False
    if hasattr(peft_model, "generation_config"):
        peft_model.generation_config.use_cache = False
    return peft_model


# --------------------------------------------------------------------------- #
# datasets                                                                    #
# --------------------------------------------------------------------------- #

def load_corpus_dataset(directory: str) -> List[str]:
    """Load every .txt file in `directory` into a flat list of strings.

    Files are loaded in deterministic (sorted) order so that re-running the
    trainer over the same corpus yields the same training distribution.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"corpus directory not found: {directory}")
    texts = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(directory, filename)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read().strip()
            if txt:
                texts.append(txt)
    log.info("loaded %d text files from %s", len(texts), directory)
    return texts


def _format_qa(question: str, answer: str, eos_marker: str = "</s>") -> str:
    """Render one QA pair as ``Q:\\nA:</s>``."""
    q = (question or "").strip()
    a = (answer or "").strip()
    if a.endswith(eos_marker):
        a = a[: -len(eos_marker)].rstrip()
    return f"Q: {q}\nA: {a}{eos_marker}"


def load_qa_dataset(directory: str, eos_marker: str = "</s>") -> List[str]:
    """Load every .json / .jsonl file in `directory` into Q:/A: strings.

    Each record must have ``"question"`` and ``"answer"`` keys. JSON
    arrays (``[{"question": ..., "answer": ...}, ...]``) and JSONL
    (one record per line) are both accepted; the file format is sniffed
    from the first non-whitespace character.

    Files are loaded in deterministic (sorted) order.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"QA dataset directory not found: {directory}")

    texts: List[str] = []
    n_files = 0
    for filename in sorted(os.listdir(directory)):
        if not (filename.endswith(".json") or filename.endswith(".jsonl")):
            continue
        path = os.path.join(directory, filename)
        n_files += 1
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(1)
            f.seek(0)
            if head == "[":
                # JSON array
                try:
                    arr = json.load(f)
                except json.JSONDecodeError as e:
                    log.warning("skipping malformed JSON file %s: %s", path, e)
                    continue
                for obj in arr:
                    if isinstance(obj, dict) and "question" in obj and "answer" in obj:
                        texts.append(_format_qa(obj["question"], obj["answer"],
                                                eos_marker))
            else:
                # JSONL
                for ln, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        log.warning("skipping bad JSONL line %s:%d: %s",
                                    path, ln, e)
                        continue
                    if isinstance(obj, dict) and "question" in obj and "answer" in obj:
                        texts.append(_format_qa(obj["question"], obj["answer"],
                                                eos_marker))
    log.info("loaded %d QA pairs from %d files in %s", len(texts), n_files, directory)
    return texts


def tokenize_and_pack(texts: List[str], tokenizer, block_size: int):
    """Tokenize, append EOS between texts, then pack into fixed-size blocks.

    Returns a HuggingFace ``datasets.Dataset`` with ``input_ids``,
    ``attention_mask``, and ``labels`` columns ready for SFTTrainer / Trainer.

    The packing intentionally drops a tail shorter than ``block_size`` so
    every block has the same shape — required for static-shape kernels.
    """
    from datasets import Dataset  # type: ignore

    if not texts:
        return Dataset.from_list([])

    enc = tokenizer(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    eos_id = tokenizer.eos_token_id
    flat_ids: List[int] = []
    for ids in enc["input_ids"]:
        flat_ids.extend(ids)
        if eos_id is not None:
            flat_ids.append(eos_id)

    usable = len(flat_ids) - (len(flat_ids) % block_size)
    flat_ids = flat_ids[:usable]
    blocks = []
    for i in range(0, usable, block_size):
        chunk = flat_ids[i : i + block_size]
        blocks.append({
            "input_ids": chunk,
            "attention_mask": [1] * len(chunk),
            "labels": list(chunk),
        })
    log.info("packed %d tokens into %d blocks of %d",
             usable, len(blocks), block_size)
    return Dataset.from_list(blocks)


# --------------------------------------------------------------------------- #
# trainer                                                                     #
# --------------------------------------------------------------------------- #

def build_trainer(peft_model, tokenizer, train_dataset, eval_dataset,
                  config: TrainingConfig):
    """Build the HF ``Trainer`` with sane defaults for LoRA + causal LM."""
    from transformers import (
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    have_eval = eval_dataset is not None and len(eval_dataset) > 0

    targs_kwargs = dict(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        lr_scheduler_type=config.lr_scheduler_type,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        bf16=(config.dtype == "bfloat16"),
        fp16=(config.dtype == "float16"),
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        report_to=config.report_to,
        remove_unused_columns=False,
        dataloader_num_workers=config.dataloader_num_workers,
        seed=config.seed,
    )
    if have_eval:
        targs_kwargs.update(
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            eval_strategy="steps",
            eval_steps=config.eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

    training_args = TrainingArguments(**targs_kwargs)

    callbacks = []
    if have_eval and config.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_threshold=config.early_stopping_threshold,
        ))

    return Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if have_eval else None,
        tokenizer=tokenizer,
        callbacks=callbacks or None,
    )


# --------------------------------------------------------------------------- #
# end-to-end driver                                                           #
# --------------------------------------------------------------------------- #

def train(config: TrainingConfig):
    """Run the full pipeline: load -> wrap -> tokenize -> train -> save.

    Returns the trained ``peft_model`` so the caller can merge or push.
    """
    os.makedirs(config.output_dir, exist_ok=True)
    log.info("training mode: %s", config.mode)

    # 1. Model + tokenizer.
    base_model, tokenizer = load_model_and_tokenizer(config)
    peft_model = apply_lora(base_model, tokenizer, config)

    # 2. Dataset.
    if config.mode == "corpus":
        raw_texts = load_corpus_dataset(config.dataset_dir)
    else:
        raw_texts = load_qa_dataset(config.dataset_dir)
    if not raw_texts:
        raise RuntimeError(f"no usable data found in {config.dataset_dir}")

    # 3. Train/eval split (deterministic: sorted load + fixed seed).
    if config.eval_split_frac > 0.0:
        from datasets import Dataset  # type: ignore
        ds = Dataset.from_list([{"text": t} for t in raw_texts])
        split = ds.train_test_split(test_size=config.eval_split_frac,
                                    seed=config.seed)
        train_texts = list(split["train"]["text"])
        eval_texts = list(split["test"]["text"])
    else:
        train_texts = raw_texts
        eval_texts = []

    train_dataset = tokenize_and_pack(train_texts, tokenizer, config.block_size)
    eval_dataset = (tokenize_and_pack(eval_texts, tokenizer, config.block_size)
                    if eval_texts else None)

    # 4. Trainer.
    trainer = build_trainer(peft_model, tokenizer, train_dataset, eval_dataset,
                            config)

    # 5. Train.
    log.info("starting training")
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    log.info("training complete; saving adapter to %s", config.output_dir)

    trainer.model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # 6. Optional push.
    if config.push_to_hub_repo:
        push_to_hub(config.output_dir, config.push_to_hub_repo,
                    private=config.push_to_hub_private)

    return trainer.model


# --------------------------------------------------------------------------- #
# push                                                                        #
# --------------------------------------------------------------------------- #

def push_to_hub(local_dir: str, repo_id: str, private: bool = True,
                commit_message: str = "Upload LoRA adapter") -> None:
    """Upload `local_dir` to a HuggingFace repo. Requires HF_TOKEN env var."""
    from huggingface_hub import HfApi, login

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=local_dir,
        repo_type="model",
        commit_message=commit_message,
    )
    log.info("pushed %s to https://huggingface.co/%s", local_dir, repo_id)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="seedsoftruth-train",
        description="Canonical LoRA trainer (continued pretraining or QA SFT).",
    )
    p.add_argument("--mode", choices=("corpus", "qa"), default="corpus",
                   help="'corpus' for raw .txt continued pretraining, "
                        "'qa' for {question,answer} JSON/JSONL fine-tuning.")
    p.add_argument("--base-model", required=True,
                   help="HF model ID or local path of the base model.")
    p.add_argument("--dataset-dir", required=True,
                   help="Directory of .txt (corpus mode) or .json/.jsonl (qa mode).")
    p.add_argument("--output-dir", required=True,
                   help="Where to save checkpoints + final adapter.")

    # Optional knobs.
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--no-4bit", action="store_true",
                   help="Disable 4-bit quantization (load in full precision).")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "auto"),
                   default="bfloat16")
    p.add_argument("--no-trust-remote-code", action="store_true")

    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--lora-target-modules", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj"])

    p.add_argument("--epochs", type=int, default=1, dest="num_train_epochs")
    p.add_argument("--batch-size", type=int, default=8,
                   dest="per_device_train_batch_size")
    p.add_argument("--grad-accum", type=int, default=2,
                   dest="gradient_accumulation_steps")
    p.add_argument("--lr", type=float, default=1e-5, dest="learning_rate")
    p.add_argument("--eval-frac", type=float, default=0.1,
                   dest="eval_split_frac",
                   help="Fraction of dataset held out for eval (0 disables).")
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--resume-from-checkpoint", default=None,
                   help="Path to a checkpoint dir to resume from.")
    p.add_argument("--push-to-hub-repo", default=None,
                   help="If set, upload the adapter to this HF repo after training.")
    p.add_argument("--public", action="store_true",
                   help="If pushing, make the HF repo public (default: private).")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.log_level)

    config = TrainingConfig(
        base_model=args.base_model,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        max_seq_length=args.max_seq_length,
        block_size=args.block_size,
        load_in_4bit=not args.no_4bit,
        dtype=args.dtype,
        trust_remote_code=not args.no_trust_remote_code,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        eval_split_frac=args.eval_split_frac,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        seed=args.seed,
        resume_from_checkpoint=args.resume_from_checkpoint,
        push_to_hub_repo=args.push_to_hub_repo,
        push_to_hub_private=not args.public,
    )

    train(config)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

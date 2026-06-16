"""
Seeds of Truth canonical LoRA training package.

Public API:

    >>> from seedsoftruth.train_ai import TrainingConfig, train
    >>> cfg = TrainingConfig(
    ...     base_model="unsloth/Qwen3-8B",
    ...     dataset_dir="./corpus",
    ...     output_dir="./lora_out",
    ...     mode="corpus",
    ... )
    >>> peft_model = train(cfg)

CLI:

    python -m seedsoftruth.train_ai \\
        --mode corpus \\
        --base-model unsloth/Qwen3-8B \\
        --dataset-dir ./corpus \\
        --output-dir ./lora_out

See README.md in this directory for the full surface and the helper
modules (``merge_lora``, ``gen_qa_pairs``, ``tokenizer_tools``).
"""

from .train import (
    TrainingConfig,
    apply_lora,
    build_trainer,
    load_corpus_dataset,
    load_model_and_tokenizer,
    load_qa_dataset,
    main,
    push_to_hub,
    tokenize_and_pack,
    train,
)

__all__ = [
    "TrainingConfig",
    "apply_lora",
    "build_trainer",
    "load_corpus_dataset",
    "load_model_and_tokenizer",
    "load_qa_dataset",
    "main",
    "push_to_hub",
    "tokenize_and_pack",
    "train",
]

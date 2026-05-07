"""
Merge a trained LoRA adapter back into its base model.

This is the standalone post-training step: given a base model
(``--base-model``, an HF repo or local path) and a LoRA adapter
(``--adapter``, also an HF repo or local path), load both, fuse the
adapter weights into the base, and save a single merged checkpoint to
``--output-dir``.

Optionally push the merged model to a HuggingFace repo with
``--push-to-hub <repo_id>``. Set ``HF_TOKEN`` in the environment first.

Run from the command line:

    python -m seedsoftruth.train_ai.merge_lora \\
        --base-model meta-llama/Llama-3.1-8B \\
        --adapter ./lora_out \\
        --output-dir ./merged_out

The script can also be invoked programmatically:

    >>> from seedsoftruth.train_ai.merge_lora import merge_lora
    >>> merge_lora(base_model="meta-llama/Llama-3.1-8B",
    ...            adapter="./lora_out",
    ...            output_dir="./merged_out")
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional

log = logging.getLogger(__name__)


def _resolve_dtype(name: str):
    import torch
    return {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(name, "auto")


def merge_lora(
    base_model: str,
    adapter: str,
    output_dir: str,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    use_8bit_base: bool = False,
    trust_remote_code: bool = True,
    push_to_hub: Optional[str] = None,
    push_private: bool = True,
    sanity_check: bool = False,
) -> None:
    """Merge `adapter` into `base_model` and save to `output_dir`.

    Parameters mirror the CLI flags. The merged model is moved to CPU
    before saving so VRAM is freed first; this is the recommended
    pattern for large models that don't fit in GPU memory twice.
    """
    import torch
    from huggingface_hub import HfApi, login
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    log.info("loading tokenizer from %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, use_fast=True, trust_remote_code=trust_remote_code
    )

    load_kwargs = dict(
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    if use_8bit_base:
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            log.info("loading base in 8-bit")
        except ImportError:
            log.warning("bitsandbytes not installed; ignoring --use-8bit-base")
    else:
        resolved = _resolve_dtype(dtype)
        if resolved != "auto":
            load_kwargs["torch_dtype"] = resolved

    log.info("loading base model %s", base_model)
    base = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)

    log.info("attaching adapter %s", adapter)
    peft_model = PeftModel.from_pretrained(base, adapter)

    log.info("merging adapter into base (this can take a while)")
    merged = peft_model.merge_and_unload()

    # Free GPU VRAM before saving — important for large models.
    merged = merged.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.makedirs(output_dir, exist_ok=True)
    log.info("saving merged model to %s", output_dir)
    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    if sanity_check:
        _sanity_check(output_dir, dtype, trust_remote_code)

    if push_to_hub:
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN env var is required to push the merged model"
            )
        api = HfApi()
        api.create_repo(push_to_hub, repo_type="model",
                        private=push_private, exist_ok=True)
        api.upload_folder(
            folder_path=output_dir,
            repo_id=push_to_hub,
            repo_type="model",
            commit_message="Add merged LoRA model",
        )
        log.info("pushed merged model to https://huggingface.co/%s", push_to_hub)


def _sanity_check(output_dir: str, dtype: str, trust_remote_code: bool) -> None:
    """Reload the merged model from disk and run one short generation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("sanity check: reloading merged model from %s", output_dir)
    tok = AutoTokenizer.from_pretrained(
        output_dir, use_fast=True, trust_remote_code=trust_remote_code
    )
    resolved = _resolve_dtype(dtype)
    model = AutoModelForCausalLM.from_pretrained(
        output_dir,
        torch_dtype=(None if resolved == "auto" else resolved),
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    prompt = "You are a helpful assistant. Briefly introduce yourself."
    inputs = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64)
    print(tok.decode(out[0], skip_special_tokens=True))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="seedsoftruth-merge-lora",
        description="Merge a trained LoRA adapter into its base model.",
    )
    p.add_argument("--base-model", required=True,
                   help="HF repo ID or local path of the base model.")
    p.add_argument("--adapter", required=True,
                   help="HF repo ID or local path of the LoRA adapter.")
    p.add_argument("--output-dir", required=True,
                   help="Where to save the merged model.")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32", "auto"),
                   default="bfloat16")
    p.add_argument("--device-map", default="auto",
                   help="HF device_map: 'auto' (default), 'cpu', or a JSON string.")
    p.add_argument("--use-8bit-base", action="store_true",
                   help="Load the base model in 8-bit to save VRAM.")
    p.add_argument("--no-trust-remote-code", action="store_true")
    p.add_argument("--push-to-hub", default=None,
                   help="If set, upload the merged model to this HF repo.")
    p.add_argument("--public", action="store_true",
                   help="If pushing, make the repo public (default: private).")
    p.add_argument("--sanity-check", action="store_true",
                   help="After saving, reload and run a short test generation.")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    merge_lora(
        base_model=args.base_model,
        adapter=args.adapter,
        output_dir=args.output_dir,
        dtype=args.dtype,
        device_map=args.device_map,
        use_8bit_base=args.use_8bit_base,
        trust_remote_code=not args.no_trust_remote_code,
        push_to_hub=args.push_to_hub,
        push_private=not args.public,
        sanity_check=args.sanity_check,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

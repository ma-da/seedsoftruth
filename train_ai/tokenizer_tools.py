"""
Small tokenizer-side utilities used around training.

This module bundles three independent tools that came up in the
training pipeline:

    add_stop_token         Append a stop marker (default ``</s>``) to
                           every non-empty line of a text file. Useful
                           when preparing a raw corpus where each line
                           is a complete document and you want EOS
                           between them at training time.

    rewrite_qa_stop_token  In a directory of ``qa_pairs_*.json`` files,
                           insert a newline before the trailing ``</s>``
                           in every answer string. Helps prevent run-on
                           sentences during inference. Only run this
                           once per corpus.

    convert_gguf_to_hf     Convert a GGUF-quantized model file back to
                           HuggingFace / PyTorch safetensors format,
                           via ``transformers``' built-in GGUF loader.

Each tool also has a ``--tool {add-stop,rewrite-qa-stop,convert-gguf}``
subcommand on the CLI:

    python -m seedsoftruth.train_ai.tokenizer_tools add-stop input.txt output.txt
    python -m seedsoftruth.train_ai.tokenizer_tools rewrite-qa-stop ./qa_dir
    python -m seedsoftruth.train_ai.tokenizer_tools convert-gguf \\
        --model-id TheBloke/Llama-2-7B-Chat-GGUF \\
        --gguf-file llama-2-7b-chat.Q5_K_M.gguf \\
        --output-dir ./hf_out
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from typing import List, Optional

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 1. add_stop_token                                                           #
# --------------------------------------------------------------------------- #

def add_stop_token(input_file: str, output_file: str,
                   stop_token: str = "</s>") -> int:
    """Read `input_file` line by line; append `stop_token` to every non-empty
    line and write the result to `output_file`. Returns the number of lines
    written.
    """
    written = 0
    with open(input_file, "r", encoding="utf-8") as src, \
         open(output_file, "w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            dst.write(f"{line} {stop_token}\n")
            written += 1
    log.info("wrote %d lines with stop token to %s", written, output_file)
    return written


# --------------------------------------------------------------------------- #
# 2. rewrite_qa_stop_token                                                    #
# --------------------------------------------------------------------------- #

def rewrite_qa_stop_token(directory: str, stop_token: str = "</s>",
                          insert: str = "\n") -> int:
    """Walk `directory` for ``*.json`` files and rewrite each entry's answer
    so that the trailing `stop_token` is preceded by `insert` (``"\\n"`` by
    default).

    Idempotency note: this function is not idempotent. Running it twice
    will produce ``\\n\\n</s>``. Run it on a fresh QA corpus only.

    Returns the number of files rewritten.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"directory not found: {directory}")

    json_files = sorted(glob.glob(os.path.join(directory, "*.json")))
    rewritten = 0
    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            log.warning("skipping malformed JSON %s: %s", path, e)
            continue

        if not isinstance(data, list):
            log.warning("skipping non-list JSON %s", path)
            continue

        changed = False
        for item in data:
            if not isinstance(item, dict):
                continue
            ans = item.get("answer")
            if isinstance(ans, str) and ans.endswith(stop_token):
                # Don't double-insert if it's already there.
                trimmed = ans[: -len(stop_token)]
                if not trimmed.endswith(insert):
                    item["answer"] = trimmed + insert + stop_token
                    changed = True

        if changed:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            log.info("rewrote %s", path)
            rewritten += 1

    log.info("rewrote %d / %d JSON files in %s",
             rewritten, len(json_files), directory)
    return rewritten


# --------------------------------------------------------------------------- #
# 3. convert_gguf_to_hf                                                       #
# --------------------------------------------------------------------------- #

def convert_gguf_to_hf(model_id: str, gguf_file: str, output_dir: str,
                       dtype: str = "float16") -> None:
    """Load a GGUF-quantized model and save it back in HuggingFace format.

    `model_id` is the HF repo (or local path) that contains the GGUF
    file; `gguf_file` is the relative filename inside that repo.

    Requires a recent enough ``transformers`` (>= 4.41) to support the
    ``gguf_file=`` argument on ``from_pretrained``.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype, torch.float16)

    log.info("loading tokenizer from %s [%s]", model_id, gguf_file)
    tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=gguf_file)

    log.info("loading GGUF weights from %s [%s]", model_id, gguf_file)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, gguf_file=gguf_file, torch_dtype=torch_dtype,
    )

    os.makedirs(output_dir, exist_ok=True)
    log.info("saving HF-format model to %s", output_dir)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="seedsoftruth-tokenizer-tools",
        description="Small utilities used around the LoRA training pipeline.",
    )
    sub = p.add_subparsers(dest="tool", required=True)

    p_add = sub.add_parser(
        "add-stop",
        help="Append a stop token to each non-empty line of a text file.",
    )
    p_add.add_argument("input_file")
    p_add.add_argument("output_file")
    p_add.add_argument("--stop-token", default="</s>")

    p_rw = sub.add_parser(
        "rewrite-qa-stop",
        help="Insert '\\n' before the trailing </s> in every answer in a "
             "directory of qa_pairs_*.json files. Only run once per corpus.",
    )
    p_rw.add_argument("directory")
    p_rw.add_argument("--stop-token", default="</s>")
    p_rw.add_argument("--insert", default="\n",
                      help=r"What to insert before the stop token (default: '\n').")

    p_gg = sub.add_parser(
        "convert-gguf",
        help="Convert a GGUF-quantized model back to HuggingFace format.",
    )
    p_gg.add_argument("--model-id", required=True,
                      help="HF repo ID or local path containing the GGUF file.")
    p_gg.add_argument("--gguf-file", required=True,
                      help="Filename of the .gguf inside the model repo.")
    p_gg.add_argument("--output-dir", required=True,
                      help="Where to write the HuggingFace-format model.")
    p_gg.add_argument("--dtype", choices=("bfloat16", "float16", "float32"),
                      default="float16")

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

    if args.tool == "add-stop":
        add_stop_token(args.input_file, args.output_file,
                       stop_token=args.stop_token)
    elif args.tool == "rewrite-qa-stop":
        rewrite_qa_stop_token(args.directory, stop_token=args.stop_token,
                              insert=args.insert)
    elif args.tool == "convert-gguf":
        convert_gguf_to_hf(args.model_id, args.gguf_file, args.output_dir,
                           dtype=args.dtype)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

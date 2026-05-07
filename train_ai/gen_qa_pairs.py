"""
Generate Q/A training pairs from a raw text corpus.

Pipeline:

    1. Walk a directory of .txt files (deterministic, sorted order).
    2. Split each file into overlapping token windows: ``overlap_chars``
       previous characters of context plus ``forward_tokens`` of new
       content. Stop tokens that appear inside the prefix window are
       respected — text before the stop token is discarded.
    3. For every window, prompt an instruction-tuned LLM (loaded via
       Unsloth or vanilla transformers) to emit 6 high-signal QA pairs
       grounded in that window.
    4. Validate each response against a strict JSON schema, drop bad
       outputs, and append the survivors to a JSONL output file with
       checkpointing every ``write_interval`` segments.

This is a cleaned-up consolidation of the
``gen_qa_pairs_beta_plus*`` scripts. All project-specific paths and
model names have been removed; everything is parameterized.

Run from the command line:

    python -m seedsoftruth.train_ai.gen_qa_pairs \\
        --model unsloth/Qwen2.5-7B-Instruct \\
        --corpus-dir ./corpus \\
        --output-dir ./qa_pairs \\
        --max-segments 1000

The script appends to ``qa_pairs_<segment_idx>.json`` files inside
``--output-dir`` so progress survives interruption. Pass ``--resume``
to pick up where the previous run left off.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional

log = logging.getLogger(__name__)


DEFAULT_STOP_MARKERS = ["</s>", "<|im_end|>", "<|endoftext|>"]

DEFAULT_SYSTEM_PROMPT = (
    "You are an analyst producing question/answer pairs that are grounded "
    "strictly in a given text segment. Your questions must be specific to "
    "the entities, events, and claims in the segment, not generic. At least "
    "three questions must share overlapping entities. Answers must be "
    "correct given the segment alone — never use outside knowledge. Output "
    "valid JSON only, no preamble, no commentary, no reasoning."
)

DEFAULT_USER_PROMPT_TEMPLATE = """Extract the 5 most important entities, terms, or claims from this segment.

<<<SEGMENT>>>
{segment}
<<<END SEGMENT>>>

Generate 6 high-signal, diverse Q-A pairs:
- 2 factual (who/what/when/where) referencing at least two of the extracted entities
- 1 explanatory (cause/effect, mechanism)
- 1 inference question grounded strictly in evidence inside the segment
- 1 domain-specific question matching the segment's domain
- 1 question that overlaps the entities of two earlier questions

Requirements:
- Every question must reference at least one extracted entity/term.
- At least 3 questions must share overlapping entities.
- Answers must be fully grounded in the given text and avoid outside knowledge.

Format as a JSON array: [{{"question": "...", "answer": "...{eos_marker}"}}, ...]"""


# --------------------------------------------------------------------------- #
# config                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class GenConfig:
    model_name: str
    corpus_dir: str
    output_dir: str

    # Window shape.
    overlap_chars: int = 200
    forward_tokens: int = 824
    stop_markers: List[str] = field(default_factory=lambda: list(DEFAULT_STOP_MARKERS))

    # Volume controls.
    max_segments_per_file: int = 500
    approx_token_budget_per_file: int = 25_000
    max_total_segments: int = 100_000
    block_limit: int = 0  # 0 = unlimited; useful for smoke testing

    # Generation settings.
    max_new_tokens: int = 1024
    temperature: float = 0.1
    top_p: float = 0.9

    # Output / checkpointing.
    write_interval: int = 50
    reporting_interval: int = 10
    resume: bool = False

    # Model load.
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    trust_remote_code: bool = True

    # The marker appended to every answer for downstream training.
    eos_marker: str = "</s>"


# --------------------------------------------------------------------------- #
# windowing                                                                   #
# --------------------------------------------------------------------------- #

def split_with_overlap_and_stops(
    text: str,
    tokenizer,
    overlap_chars: int = 200,
    forward_tokens: int = 824,
    stop_markers: Optional[List[str]] = None,
    max_segments: Optional[int] = None,
    approx_token_budget: Optional[int] = None,
) -> List[str]:
    """Produce overlapping windows of `text`.

    Each window is ``overlap_chars`` characters of previous context plus
    ``forward_tokens`` newly-decoded tokens. If a stop marker appears
    inside the previous-context window, only the text *after* the stop
    marker is kept — this prevents bleed-through across document
    boundaries that have been marked with EOS.
    """
    stop_markers = stop_markers or list(DEFAULT_STOP_MARKERS)
    full_ids = tokenizer.encode(text, add_special_tokens=False)
    segments: List[str] = []
    total_tokens_used = 0
    pos = 0

    while pos < len(full_ids):
        if approx_token_budget and total_tokens_used >= approx_token_budget:
            break
        if max_segments and len(segments) >= max_segments:
            break

        # Previous-context window in raw chars.
        if pos == 0:
            prev_text = ""
        else:
            back_start = max(0, pos - 2000)
            raw_back = tokenizer.decode(full_ids[back_start:pos],
                                        skip_special_tokens=False)
            prev_text = raw_back[-overlap_chars:]

        # Trim previous window at the last stop marker, if any.
        last_stop_index = -1
        for m in stop_markers:
            idx = prev_text.rfind(m)
            if idx != -1:
                last_stop_index = max(last_stop_index, idx + len(m))
        if last_stop_index != -1:
            prev_text = prev_text[last_stop_index:]

        next_end = min(pos + forward_tokens, len(full_ids))
        new_chunk_text = tokenizer.decode(full_ids[pos:next_end],
                                          skip_special_tokens=False)
        combined = (prev_text + new_chunk_text).strip()
        if combined:
            segments.append(combined)

        total_tokens_used += (next_end - pos)
        pos = next_end

    return segments


# --------------------------------------------------------------------------- #
# response validation                                                         #
# --------------------------------------------------------------------------- #

_RESPONSE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "answer": {"type": "string"},
        },
        "required": ["question", "answer"],
    },
}


def _validate_qa_response(text: str) -> Optional[list]:
    """Try to parse `text` as a JSON array of {question, answer} dicts.

    If the optional ``jsonschema`` package is installed, the array is
    schema-validated. Returns the parsed list on success, None on failure.
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, list):
        return None
    try:
        from jsonschema import validate as _validate, ValidationError
        try:
            _validate(instance=data, schema=_RESPONSE_SCHEMA)
        except ValidationError:
            return None
    except ImportError:
        # Soft check without jsonschema.
        for item in data:
            if not (isinstance(item, dict)
                    and "question" in item and "answer" in item):
                return None
    return data


def _extract_json_array(response: str) -> Optional[str]:
    """Find the outermost JSON array in `response` and return it."""
    start = response.find("[")
    if start == -1:
        return None
    end = response.rfind("]")
    if end <= start:
        return None
    return response[start:end + 1]


# --------------------------------------------------------------------------- #
# model load                                                                  #
# --------------------------------------------------------------------------- #

def _load_model_for_generation(config: GenConfig):
    """Load a model + tokenizer set up for inference.

    Tries Unsloth first (faster); falls back to vanilla transformers.
    """
    try:
        from unsloth import FastLanguageModel  # type: ignore
        log.info("loading generator via Unsloth: %s", config.model_name)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit,
            device_map="auto",
        )
        model = FastLanguageModel.for_inference(model)
        return model, tokenizer
    except ImportError:
        log.warning("Unsloth not installed; falling back to transformers")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("loading generator via transformers: %s", config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, use_fast=True,
        trust_remote_code=config.trust_remote_code,
    )
    load_kwargs = dict(
        trust_remote_code=config.trust_remote_code,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if config.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        except ImportError:
            log.warning("bitsandbytes not installed; loading in fp16/bf16")
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **load_kwargs)
    model.eval()
    return model, tokenizer


def _ensure_pad_eos(tokenizer, model) -> None:
    """Make sure tokenizer.eos_token / pad_token / their IDs are consistent."""
    added = False
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
        added = True
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        added = True
    if added and hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))


# --------------------------------------------------------------------------- #
# corpus loading                                                              #
# --------------------------------------------------------------------------- #

def load_corpus_texts(directory: str) -> List[str]:
    """Load every .txt file in `directory` (sorted)."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"corpus directory not found: {directory}")
    txt_files = sorted(
        f for f in os.listdir(directory) if f.lower().endswith(".txt")
    )
    texts: List[str] = []
    for filename in txt_files:
        path = os.path.join(directory, filename)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read().strip()
            if txt:
                texts.append(txt)
    log.info("loaded %d text files from %s", len(texts), directory)
    return texts


# --------------------------------------------------------------------------- #
# main loop                                                                   #
# --------------------------------------------------------------------------- #

def _resume_index(output_dir: str) -> int:
    """Return the next segment index to process based on existing output files."""
    files = glob.glob(os.path.join(output_dir, "qa_pairs_*.json"))
    if not files:
        return 1
    nums = []
    for f in files:
        m = re.search(r"qa_pairs_(\d+)\.json$", f)
        if m:
            try:
                nums.append(int(m.group(1)))
            except ValueError:
                continue
    return (max(nums) + 1) if nums else 1


def generate_qa_pairs(config: GenConfig) -> int:
    """Run the full pipeline. Returns total segments processed in this run."""
    import torch  # noqa: F401  # required for cuda.empty_cache below

    os.makedirs(config.output_dir, exist_ok=True)

    model, tokenizer = _load_model_for_generation(config)
    _ensure_pad_eos(tokenizer, model)

    # 1. Build the segment list.
    raw_texts = load_corpus_texts(config.corpus_dir)
    blocks: List[str] = []
    for text in raw_texts:
        segments = split_with_overlap_and_stops(
            text,
            tokenizer,
            overlap_chars=config.overlap_chars,
            forward_tokens=config.forward_tokens,
            stop_markers=config.stop_markers,
            max_segments=config.max_segments_per_file,
            approx_token_budget=config.approx_token_budget_per_file,
        )
        blocks.extend(segments)
        if config.block_limit and len(blocks) >= config.block_limit:
            blocks = blocks[: config.block_limit]
            log.info("block_limit reached (%d); stopping segmentation",
                     config.block_limit)
            break
    log.info("prepared %d segments", len(blocks))

    # 2. Resume to the right starting index, if requested.
    start_i = _resume_index(config.output_dir) if config.resume else 1
    if config.resume:
        log.info("resuming from segment index %d", start_i)

    # 3. Iterate.
    qa_pairs_buffer: list = []
    total_pairs_written = max(0, start_i - 1)
    last_idx = start_i - 1

    has_chat_template = getattr(tokenizer, "chat_template", None) not in (None, "")

    for idx, segment in enumerate(blocks[start_i - 1:], start=start_i):
        if total_pairs_written > config.max_total_segments:
            log.info("max_total_segments hit (%d); stopping",
                     config.max_total_segments)
            break

        last_idx = idx
        if idx % config.reporting_interval == 0:
            log.info("processing segment %d / %d", idx, len(blocks))

        user_prompt = DEFAULT_USER_PROMPT_TEMPLATE.format(
            segment=segment, eos_marker=config.eos_marker,
        )
        try:
            messages = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            if has_chat_template:
                enc = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                if not isinstance(enc, dict):
                    enc = {"input_ids": enc}
                inputs = {k: v.to(model.device) for k, v in enc.items()}
                if "attention_mask" not in inputs:
                    import torch
                    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
            else:
                prompt_str = (
                    f"### System:\n{DEFAULT_SYSTEM_PROMPT}\n\n"
                    f"### User:\n{user_prompt}\n\n"
                    f"### Assistant:\n"
                )
                enc = tokenizer(prompt_str, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in enc.items()}

            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=True,
                num_return_sequences=1,
                temperature=config.temperature,
                top_p=config.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            snippet = _extract_json_array(response)
            if not snippet:
                log.debug("segment %d: no JSON array found in response", idx)
                continue
            parsed = _validate_qa_response(snippet)
            if parsed is None:
                log.debug("segment %d: response failed validation", idx)
                continue

            for qa in parsed:
                if not qa["answer"].endswith(config.eos_marker):
                    qa["answer"] += config.eos_marker
                qa_pairs_buffer.append(qa)
            total_pairs_written += 1

        except Exception as e:
            log.warning("segment %d failed: %s", idx, e)
            continue
        finally:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        # Periodic flush.
        if idx > 0 and idx % config.write_interval == 0 and qa_pairs_buffer:
            out_path = os.path.join(config.output_dir, f"qa_pairs_{idx}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(qa_pairs_buffer, f, ensure_ascii=False, indent=2)
            log.info("wrote %d QA pairs to %s", len(qa_pairs_buffer), out_path)
            qa_pairs_buffer = []

    # Final flush.
    if qa_pairs_buffer:
        out_path = os.path.join(config.output_dir, f"qa_pairs_{last_idx}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(qa_pairs_buffer, f, ensure_ascii=False, indent=2)
        log.info("final flush: wrote %d QA pairs to %s",
                 len(qa_pairs_buffer), out_path)

    log.info("generated %d total segments worth of QA pairs",
             total_pairs_written)
    return total_pairs_written


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="seedsoftruth-gen-qa-pairs",
        description="Generate QA training pairs from a raw text corpus.",
    )
    p.add_argument("--model", required=True, dest="model_name",
                   help="HF model ID or local path of an instruction-tuned LLM.")
    p.add_argument("--corpus-dir", required=True,
                   help="Directory of .txt files to segment.")
    p.add_argument("--output-dir", required=True,
                   help="Where to write qa_pairs_<idx>.json files.")
    p.add_argument("--overlap-chars", type=int, default=200)
    p.add_argument("--forward-tokens", type=int, default=824)
    p.add_argument("--max-segments-per-file", type=int, default=500)
    p.add_argument("--approx-token-budget-per-file", type=int, default=25_000)
    p.add_argument("--max-total-segments", type=int, default=100_000)
    p.add_argument("--block-limit", type=int, default=0,
                   help="Cap total blocks for smoke-testing (0 = unlimited).")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--write-interval", type=int, default=50)
    p.add_argument("--reporting-interval", type=int, default=10)
    p.add_argument("--resume", action="store_true",
                   help="Resume from the highest existing qa_pairs_<idx>.json.")
    p.add_argument("--eos-marker", default="</s>")
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--no-trust-remote-code", action="store_true")
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

    config = GenConfig(
        model_name=args.model_name,
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        overlap_chars=args.overlap_chars,
        forward_tokens=args.forward_tokens,
        max_segments_per_file=args.max_segments_per_file,
        approx_token_budget_per_file=args.approx_token_budget_per_file,
        max_total_segments=args.max_total_segments,
        block_limit=args.block_limit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        write_interval=args.write_interval,
        reporting_interval=args.reporting_interval,
        resume=args.resume,
        eos_marker=args.eos_marker,
        load_in_4bit=not args.no_4bit,
        trust_remote_code=not args.no_trust_remote_code,
    )
    generate_qa_pairs(config)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

# Stage 4 — Fine-tuning the Language Model (LoRA)

This stage produces or refreshes the **LoRA adapter** that gets loaded
on top of Qwen2.5-7B-Instruct in the production HF Inference Endpoint.
It is the rarest stage in the pipeline — typically run every few months
when the team decides the model needs to know more recent vocabulary or
needs its tone re-calibrated.

> **First-time reader: you probably don't need to run this.** Adding
> new content to the corpus only requires stages 1–3 (crawl, clean,
> ingest) plus a Flask restart. The model retrieves the new chunks at
> query time and grounds its answer in them. Fine-tuning is for changing
> *how* the model talks, not *what facts* it has access to. See the
> overview's [fine-tuning vs RAG section](00-overview.md#why-fine-tuning-alone-isnt-enough-and-why-we-need-rag).

## Conceptual background

### Why fine-tune at all?

The base model (Qwen2.5-7B-Instruct) is already excellent at general
question-answering. What it doesn't know is:

- **House style.** How the team wants citations formatted, what hedging
  language to use on contested topics, when to say "I don't have
  enough information" vs when to speculate.
- **Domain vocabulary in the right register.** The base model has read
  about Iran-Contra, but it has not read PEERS' specific framing of
  these events, the particular set of source authors PEERS cites, the
  internal coherence of how these topics are discussed within the
  community.
- **Refusal calibration.** Off-the-shelf instruction-tuned models
  often refuse questions about contested topics ("I can't speculate
  about that"). PEERS wants the model to engage thoughtfully when the
  retrieval finds substantive sources, and decline only when retrieval
  comes back empty.

Fine-tuning teaches these *behaviors* without trying to inject the
factual content of the corpus into the weights themselves. The corpus
content stays in the retrieval DB.

### What is LoRA?

A 7-billion-parameter model has ~14 GB of weight tensors in fp16. Full
fine-tuning means updating all of them, which requires hundreds of GB
of GPU memory (for the model + optimizer state + gradients) and
produces a new 14-GB file per training run.

**LoRA (Low-Rank Adaptation)** observes that most useful fine-tuning
adjustments can be expressed as a *small* delta to the original
weights — specifically, a sum of low-rank matrix products. Instead of
training all the weights, you train a small adapter (typically 10–100
MB) that gets *added* to specific weight matrices at inference time.

Concretely, where the base model has a weight matrix W, LoRA learns two
small matrices A and B such that:

```
W_effective = W + (B @ A) * alpha
```

with rank(A) = rank(B) = r << dim(W). For Qwen-7B, typical settings
are r=8 or r=16. The trainable parameters are ~0.1% of the base model.

Benefits:
- Trains on a single 24–48 GB GPU (an A100, an A6000, or a Colab Pro+
  L4).
- Produces a small file (~50–200 MB) that ships separately from the
  base model.
- Multiple LoRAs can be A/B-tested without re-hosting the 14-GB base.

Costs:
- Quality plateaus earlier than full fine-tuning at very large dataset
  sizes (not a concern for this project's scale).
- Adapter has to be loaded at inference time, which is one extra
  config setting in the HF endpoint.

### What does the training dataset look like?

The model is trained on **instruction–response pairs**:

```json
{"instruction": "What did the Wall Street Journal report about Fred Burks?",
 "response":    "The Wall Street Journal's February 26, 2005 front-page article
                 'Lost in Translation' described how Fred Burks, a U.S.
                 government interpreter, went on to publish information about
                 government activity he witnessed firsthand..."}
```

Many thousands of these pairs are auto-generated from the corpus by
the scripts in `train_ai/gen_qa_pairs_*.py`, then optionally
hand-curated. The generation pipeline:

1. Pick a chunk from the corpus DB.
2. Prompt a stronger LLM (Qwen-32B or similar via DeepInfra) to
   produce 2–4 question/answer pairs grounded in that chunk.
3. Validate the pairs (do they reference the source? does the answer
   span exist in the source text?).
4. Discard pairs that look hallucinated; keep the rest.

This is sometimes called *synthetic data generation* or *self-instruct*.
The team has produced several corpora this way:
`corpus_beta_plus`, `corpus_beta_plus_overlaps`, etc. — each is a
JSONL file of instruction/response pairs.

## The code

Everything lives under `/Volumes/SSK/peers_dev/train_ai/`. Notable files:

```
train_ai/
├── gen_qa_pairs.ipynb               Original Jupyter notebook — dataset gen
├── gen_qa_pairs_beta_plus.py        Script versions for batch runs
├── gen_qa_pairs_beta_plus2.py
├── gen_qa_pairs_beta_plus_overlaps.py
├── gen_qa_pairs_beta_plus_overlaps2.py
│
├── handler.py                       HF Inference Endpoint handler (production)
├── handler_basic.py                 Simpler reference handler
├── handler_v4a.py                   Recent variant
│
├── inference_test_*.ipynb           Eval notebooks — sanity-check a fresh
│                                    LoRA against held-out questions
├── inference_test_lora_qwen3_v4_qa2.ipynb   Most recent (v4 QA-pair LoRA)
│
├── convert_gguf_to_hf.py            Converts a llama.cpp-style adapter to
│                                    HF format for deployment
└── add_stop_token.py                One-time tokenizer surgery for stop-tokens
```

The actual training is done in notebooks (the team's preference for
this stage), typically using the **Unsloth** library on top of
HuggingFace `transformers` and `peft`. Unsloth is a wrapper that gives
significant speedups over vanilla HF training loops — roughly 2x
faster with the same memory footprint.

The default training recipe (from `gen_qa_pairs_beta_plus.py` and the
team's notebooks):

- **Base model:** `unsloth/Qwen2.5-7B-Instruct` (4-bit quantized via
  `load_in_4bit=True`).
- **Adapter rank:** typically `r=8` to `r=16`.
- **Sequence length:** 1024 tokens.
- **Learning rate:** in the 1e-4 to 2e-4 range with cosine schedule.
- **Batch size:** small (1–4) with gradient accumulation 4–16 to
  reach effective batches of 8–64.
- **Epochs:** 1–3 depending on dataset size.

The full training script will load the base model in 4-bit (so it fits
on a 24 GB GPU), apply the LoRA modules to the attention projections,
train, then save the adapter via `model.save_pretrained()`. The
result is a folder like:

```
output/
├── adapter_config.json
├── adapter_model.safetensors
└── tokenizer.json
```

That folder is what gets deployed to the HF Inference Endpoint.

## Deploying a new LoRA

The production deployment is a **Hugging Face Inference Endpoint** —
HF's managed-inference product that runs a single model behind an
HTTPS API. The team's endpoint loads Qwen2.5-7B-Instruct as the base
model and the project's LoRA adapter on top.

To deploy a new adapter:

1. Train the adapter (notebook).
2. Test it locally with `inference_test_lora_qwen3_v4_qa2.ipynb`
   against held-out questions.
3. Upload the adapter folder to a private HF repo.
4. Update the endpoint config to point at the new adapter (HF UI or
   API).
5. Rolling restart of the endpoint (HF handles this).
6. On the Seeds of Truth Flask side, no change is needed — the
   `HFEndpointLLM` adapter in `model_adapters.py` just calls whichever
   endpoint URL is configured. The model behind that URL switches
   silently.

## Time and cost reality check

For a typical refresh:

- **Dataset generation** (10k Q/A pairs from the corpus): ~$30–80 of
  DeepInfra/OpenAI API spend, ~1 day of wall time.
- **Training a single LoRA** on a 24-GB GPU: 4–12 hours.
- **Iterating** (try different ranks, dataset mixes, learning rates):
  multiply by 4–8.
- **HF Inference Endpoint hosting**: ~$1/hour for a small instance,
  $5–10/hour for an A100. Production is typically the small instance.

A full refresh end-to-end is a multi-day affair. **Don't think of
fine-tuning as the answer to "the system doesn't know about [recent
event]" — that's what RAG is for.** Fine-tuning is the answer when the
team observes the model handling a class of questions in a
systematically wrong way (e.g., refusing to engage, hedging too much,
hedging too little) and wants to recalibrate that behavior.

## What we don't cover here

This document is a conceptual primer plus a pointer at where the code
lives. The actual training notebooks contain the working recipe and
the most recent dataset-mix decisions; defer to them for any specific
change. If you're not already comfortable with HuggingFace + PEFT +
LoRA, the [Unsloth documentation](https://github.com/unslothai/unsloth)
is the gentlest starting point.

## Where to go next

[05-serving.md](05-serving.md) covers how the trained model gets
called at query time and how retrieval results are assembled into the
prompt.

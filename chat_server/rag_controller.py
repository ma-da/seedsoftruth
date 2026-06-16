"""
rag_controller.py (SQLite hybrid edition)

Public Flask-facing interface: boot(), search_references(), ask(),
ask_model_only(), plus model-readiness and job-queue helpers.

As of the 2026-05-20 refactor the retrieval engine lives in
``rag_retrieval.py`` and context assembly in ``rag_context.py``. This
module owns model-adapter orchestration and the in-process job queue, and
re-exports the retrieval/context entrypoints below so existing
``rag_controller.<name>`` call sites keep working unchanged.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from dataclasses_json import dataclass_json

import logging_config
import model_adapters
import utils
from rag_context import build_context_v2  # noqa: F401

# Retrieval + context engines. Imported names are re-exported as part of
# this module's public API (rag_controller.search_references, .boot, ...).
from rag_retrieval import (  # noqa: F401
    ENABLE_MIN_GATING,
    HYBRID_DB_PATH,
    MIN_GATE_SCORE_FLOOR,
    RetrievalState,
    _detect_fts_branch,
    _hybrid_search_sqlite5,
    _min_gate,
    boot,
    clean_rag_references,
    clean_retrieval_text,
    search_references,
    truncate_question,
)

rag_logger = logging_config.get_logger("rag")

COPYRIGHT_BLOCK_MSG = "This text protected by copyright."

# ------------------ MODEL ADAPTERS ------------------
hf_llm_model = model_adapters.LLMFactory.create("hf")
deep_infra_llm_model = model_adapters.LLMFactory.create("deepinfra")
spark_llm_model = model_adapters.LLMFactory.create("spark")
sim_model = model_adapters.LLMFactory.create("sim")
vllm_llm_model = model_adapters.LLMFactory.create("vllm")

DEFAULT_MODEL_TYPE = os.getenv("MODEL_ADAPTER", "hf").strip().lower()

if DEFAULT_MODEL_TYPE == "spark":
    llm_model = spark_llm_model
elif DEFAULT_MODEL_TYPE == "deepinfra":
    llm_model = deep_infra_llm_model
elif DEFAULT_MODEL_TYPE == "vllm":
    llm_model = vllm_llm_model
else:
    llm_model = hf_llm_model

llm_models = [hf_llm_model, deep_infra_llm_model, spark_llm_model, vllm_llm_model]


def get_model_type(type: str) -> model_adapters.LLMStrategy:
    """Resolves a model-type name to its LLM adapter instance.

    Args:
        type: A model-type name. ``"default"`` returns the process default
            adapter; otherwise one of the names accepted by
            ``model_adapters.is_valid_model_type`` (``"hf"``,
            ``"deepinfra"``, ``"spark"``, ``"vllm"``, ``"sim"``).

    Returns:
        The matching ``model_adapters.LLMStrategy`` adapter.

    Raises:
        ValueError: If ``type`` is not a recognized model type.
    """
    global llm_model, hf_llm_model, deep_infra_llm_model, spark_llm_model, sim_model, vllm_llm_model

    if type == "default":
        return llm_model

    if not model_adapters.is_valid_model_type(type):
        raise ValueError(f"Invalid model type: {type}")

    if type == "hf":
        return hf_llm_model
    elif type == "deepinfra":
        return deep_infra_llm_model
    elif type == "spark":
        return spark_llm_model
    elif type == "sim":
        return sim_model
    elif type == "vllm":
        return vllm_llm_model
    else:
        raise ValueError(f"Unknown model type: {type}")


rag_logger.info(f"DEFAULT_MODEL_TYPE={DEFAULT_MODEL_TYPE}")
rag_logger.info(f"llm_model selected: {llm_model.name()}")
rag_logger.info(f"available model names: {[m.name() for m in llm_models]}")


@dataclass
class QueuedJob:
    """An inbound generation request waiting in the in-process job queue.

    Attributes:
        user_id: Identifier of the requesting user.
        job_id: Unique identifier for this job.
        model_type: Name of the model adapter to use.
        prompt: The user's question/prompt text.
        subsets: Optional list of corpus subsets to restrict retrieval to.
        rag_algo_choice: Selector for the retrieval algorithm variant.
    """

    user_id: str
    job_id: str
    model_type: str
    prompt: str
    subsets: List[str] | None = None
    rag_algo_choice: int = 5
    # Added 2026-05-26 in async-chat migration: previously the synchronous
    # /api/chat path consumed these from the request, but the queued path
    # silently dropped them. With queued now the only path, missing them
    # here meant a user who selected V2 prompt or toggled RAG off was
    # quietly given V1 + RAG-on. Defaults match chat_with_corpus.
    use_rag: bool = True
    prompt_type: int = 0  # 0-indexed internal value (1-indexed on the wire)
    # Wall-clock seconds since epoch, set when the job is appended to the
    # in-memory queue. Used by worker_body to compute queue-wait latency
    # (time spent waiting for the worker to pick it up) separately from
    # the chat_with_corpus processing time.
    queued_at: float = 0.0


# ------------------ HF ORCHESTRATION ------------------


def is_model_ready(timeout=model_adapters.MODEL_TIMEOUT_SECS) -> bool:
    """Checks whether the process default model is ready to serve requests.

    Args:
        timeout: Seconds to wait for the readiness check.

    Returns:
        True if the default model responded ready within the timeout.
    """
    return utils.do_async_to_sync(
        lambda: llm_model.is_model_ready(timeout=timeout)
    )()


def is_model_type_ready(
    model_type: str, timeout=model_adapters.MODEL_TIMEOUT_SECS
) -> bool:
    """Checks whether a specific model type is ready to serve requests.

    Args:
        model_type: The model-type name to resolve and probe.
        timeout: Seconds to wait for the readiness check.

    Returns:
        True if that model responded ready within the timeout.
    """
    model = get_model_type(model_type)
    return utils.do_async_to_sync(
        lambda: model.is_model_ready(timeout=timeout)
    )()


def send_warmup() -> bool:
    """Sends a warm-up request to the process default model.

    Returns:
        True if the warm-up request succeeded.
    """
    return utils.do_async_to_sync(lambda: llm_model.send_warmup())()


def send_warmup_for(model_type: str) -> bool:
    """Sends a warm-up request to a specific model type.

    Args:
        model_type: The model-type name to resolve and warm up.

    Returns:
        True if the warm-up request succeeded.
    """
    model = get_model_type(model_type)
    return utils.do_async_to_sync(lambda: model.send_warmup())()


async def ask(
    model_adaptor: model_adapters.LLMStrategy,
    state: RetrievalState,
    question: str,
    *,
    context_k: int = 10,
    top_k: int = 10,
    verbose: bool = True,
    use_double_prompt: bool = False,
    subsets: Optional[List[str]] = None,
    rag_algo_choice: int = 5,
    prompt_type: int = 0,
) -> str:
    """Answers a question with retrieval-augmented generation.

    Truncates the question if needed, retrieves supporting references via
    ``search_references``, assembles a context block with
    ``build_context_v2``, builds the full prompt (system prompt + question
    block + optional documents block), and calls the model adapter to
    generate the answer. The documents block is omitted entirely when
    retrieval produced no usable context.

    Args:
        model_adaptor: The LLM adapter used to generate the answer.
        state: The retrieval engine state (index, DB connection, etc.).
        question: The user's question.
        context_k: Maximum number of retrieved docs fed to context
            assembly.
        top_k: Maximum number of references to retrieve.
        verbose: When True, logs token-count diagnostics and the prompt.
        use_double_prompt: When True, prepends a self-restatement
            instruction asking the model to reframe the question and
            identify strong evidence before answering.
        subsets: Optional list of corpus subsets to restrict retrieval to.
        rag_algo_choice: Selector for the retrieval algorithm variant.
        prompt_type: Index selecting which system prompt to use.

    Returns:
        The model's answer, stripped, prefixed with a truncation notice if
        the question had to be shortened.
    """
    model_adaptor_name = model_adaptor.name()
    rag_logger.info(f"ask() using '{model_adaptor_name}': {question[:80]}...")

    q, truncated = truncate_question(question)

    t0 = time.time()
    refs = await search_references(
        state,
        q,
        top_k=top_k,
        verbose=verbose,
        subsets=subsets,
        rag_algo_choice=rag_algo_choice,
    )
    docs = refs.get("results", [])
    # build_context_v2: stricter filter + canonical-entity awareness — see
    # the "Who killed JFK?" -> JFK-Airport snippets failure that motivated
    # this change. Pass `state` so the entity disambiguation path is active.
    context = build_context_v2(docs, q, state=state, context_k=context_k)
    rag_logger.info(f"chat search_references results: {docs}")

    system_prompt = model_adapters.get_system_prompt(prompt_type)
    rag_logger.info(f"system prompt type: {prompt_type}")

    # Self-restatement variant: ask the model to reframe before answering. This
    # is a real prompting technique (improves coherence on ambiguous questions)
    # and replaces the prior "double prompt" mode that just typed the question
    # twice verbatim with no upside.
    if use_double_prompt:
        question_block = (
            "Agent Question:\n"
            f"{q}\n\n"
            "Before answering: restate the question in your own words and "
            "identify what would constitute strong evidence. Then answer.\n\n"
        )
    else:
        question_block = "Agent Question:\n" f"{q}\n\n"

    # DOCUMENTS section is included only when retrieval produced something.
    # When `context` is empty (e.g., v4's min-gate declined), no preamble is
    # added at all — the model just sees the system prompt + question and
    # is free to answer from general knowledge per the system prompt's rules.
    if str(context).strip():
        docs_block = (
            "Documents (use if helpful, otherwise ignore):\n" f"{context}"
        )
    else:
        docs_block = ""

    # The USER content is the question + (optional) document context.
    # The system_prompt is NOT prepended here — we pass it to generate()
    # as a separate kwarg. OpenAI-shape adapters (Spark, vLLM) place it
    # in the `system` role of the messages array; single-input adapters
    # (HF, DeepInfra) prepend it themselves. Prepending in both places
    # would duplicate the system content on the wire — see the
    # 2026-05-29 bugfix that motivated this split.
    user_content = f"{question_block}{docs_block}"

    prompt_tokens_len = None
    # Estimate tokens against the FULL composed prompt (system + user)
    # so the logged total matches what the model actually sees end to
    # end across both adapter shapes.
    full_for_estimate = f"{system_prompt}\n\n{user_content}"
    if verbose:
        prompt_tokens_len = utils.estimate_tokens(full_for_estimate)
        rag_logger.info(f"Retrieved context in {time.time() - t0:.2f}s")

    if verbose:
        context_tokens_len = utils.estimate_tokens(context)
        total_tokens_len = prompt_tokens_len + context_tokens_len

        rag_logger.info(
            f"prompt_total_toks {total_tokens_len}, prompt_toks {prompt_tokens_len}, context_toks {context_tokens_len}"
        )

    rag_logger.info(f"-- PROMPT --\n{full_for_estimate} \n-- END PROMPT --")

    answer = model_adaptor.generate(
        user_content,
        system_prompt=system_prompt,
        temperature=model_adapters.DEFAULT_TEMPERATURE,
        max_new_tokens=model_adapters.DEFAULT_MAX_TOKENS,
    )

    if truncated:
        answer = "(Question truncated)\n\n" + answer

    return answer.strip()


async def ask_model_only(
    model_adaptor: model_adapters.LLMStrategy,
    state: RetrievalState,
    question: str,
    *,
    verbose: bool = True,
    use_double_prompt: bool = False,
    prompt_type: int = 0,
) -> str:
    """Answers a question with the model alone, skipping retrieval.

    Truncates the question if needed, builds a prompt from the system
    prompt and the question (with no retrieved documents), and calls the
    model adapter to generate the answer. Used as a no-RAG baseline.

    Args:
        model_adaptor: The LLM adapter used to generate the answer.
        state: The retrieval engine state. Accepted for signature parity
            with ``ask``; not used for retrieval here.
        question: The user's question.
        verbose: When True, logs the prompt a second time for diagnostics.
        use_double_prompt: When True, repeats the question line twice in
            the prompt.
        prompt_type: Index selecting which system prompt to use.

    Returns:
        The model's answer, stripped, prefixed with a truncation notice if
        the question had to be shortened.
    """
    model_adaptor_name = model_adaptor.name()
    rag_logger.info(
        f"ask_model_only() using '{model_adaptor_name}': {question[:80]}..."
    )

    q, truncated = truncate_question(question)

    system_prompt = model_adapters.get_system_prompt(prompt_type)
    rag_logger.info(f"system prompt type: {prompt_type}")

    # USER content only (no system prompt prefix) — same split as ask().
    if use_double_prompt:
        user_content = (
            "Agent Question:\n"
            f"{q}\n"
            f"{q}\n"
        )
    else:
        user_content = (
            "Agent Question:\n"
            f"{q}\n"
        )

    full_for_log = f"{system_prompt}\n\n{user_content}"
    rag_logger.info(f"-- PROMPT --\n{full_for_log} \n-- END PROMPT --")

    if verbose:
        rag_logger.info(
            f"-- PROMPT (model-only) --\n{full_for_log}\n-- END PROMPT --"
        )

    answer = model_adaptor.generate(
        user_content,
        system_prompt=system_prompt,
        temperature=model_adapters.DEFAULT_TEMPERATURE,
        max_new_tokens=model_adapters.DEFAULT_MAX_TOKENS,
    )

    if truncated:
        answer = "(Question truncated)\n\n" + answer

    return str(answer or "").strip()


# ------------------ QUEUEING ------------------

job_queue: List[QueuedJob] = []
job_lock = threading.Lock()

inflight_users = {}
inflight_lock = threading.Lock()


def queue_job(
    user_id: str,
    job_id: str,
    model_type: str,
    prompt: str,
    subsets: Optional[List[str]],
    rag_algo_choice: int = 5,
    use_rag: bool = True,
    prompt_type: int = 0,
) -> int:
    """Appends a new generation request to the inbound job queue.

    Wraps the arguments in a ``QueuedJob`` (stamping ``queued_at`` with the
    current time) and enqueues it under the job lock.

    Args:
        user_id: Identifier of the requesting user.
        job_id: Unique identifier for this job.
        model_type: Name of the model adapter to use.
        prompt: The user's question/prompt text.
        subsets: Optional list of corpus subsets to restrict retrieval to.
        rag_algo_choice: Selector for the retrieval algorithm variant.
        use_rag: Whether to use retrieval augmentation.
        prompt_type: Index selecting which system prompt to use.

    Returns:
        The job queue depth after the job was appended.
    """
    queued_job = QueuedJob(
        user_id, job_id, model_type, prompt, subsets, rag_algo_choice,
        use_rag, prompt_type,
        queued_at=time.time(),
    )
    with job_lock:
        job_queue.append(queued_job)
        return len(job_queue)


def fetch_queued_job_info(user_id: str) -> Tuple[int, Optional[QueuedJob]]:
    """Finds the first queued job belonging to a user.

    Args:
        user_id: Identifier of the user to look up.

    Returns:
        A ``(index, job)`` tuple for the user's first queued job, or
        ``(0, None)`` if the user has no job in the queue.
    """
    with job_lock:
        for i, item in enumerate(job_queue):
            if user_id == item.user_id:
                return i, item
    return 0, None


def queue_position_for_job(job_id: str) -> int:
    """Return the 1-indexed position of ``job_id`` in the in-memory queue,
    or 0 if the job isn't in the queue right now.

    Used by ``/api/job/<id>`` to surface "queued, position N" to a polling
    client without making that client own its own queue model.

    Position 0 means either (a) the worker has already picked the job up
    (it's now 'processing') but the DB transition hasn't landed yet, or
    (b) the job genuinely isn't in the in-mem queue (rare race). The
    caller treats 0 as "unknown".
    """
    with job_lock:
        for i, item in enumerate(job_queue):
            if item.job_id == job_id:
                return i + 1
    return 0


def has_queued_job() -> bool:
    """Reports whether the inbound job queue is non-empty.

    Returns:
        True if at least one job is queued.
    """
    with job_lock:
        return len(job_queue) > 0


def get_next_queued_job() -> Optional[QueuedJob]:
    """Returns the job at the head of the queue without removing it.

    Returns:
        The next ``QueuedJob``, or None if the queue is empty.
    """
    with job_lock:
        if len(job_queue) == 0:
            return None
        return job_queue[0]


def pop_next_queued_job() -> None:
    """Removes the job at the head of the queue.

    Does nothing if the queue is already empty.
    """
    with job_lock:
        if len(job_queue) == 0:
            return
        job_queue.pop(0)


def job_queue_len() -> int:
    """Returns the current depth of the inbound job queue.

    Returns:
        The number of queued jobs.
    """
    with job_lock:
        return len(job_queue)


# ------------------ MAIN ------------------


async def main() -> None:
    """Boots the retrieval engine and runs a sample reference search.

    Provides a simple manual smoke test when the module is run directly.
    """
    state = boot()
    refs = await search_references(state, "death squads in Haiti", top_k=10)
    rag_logger.info(json.dumps(refs, indent=2, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    asyncio.run(main())

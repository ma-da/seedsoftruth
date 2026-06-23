"""Retrieval and chat operations over the corpus.

Thin orchestration around ``rag_controller``: retrieval-only search,
model-only answers, and full retrieval-augmented chat against the shared
``RetrievalState`` from :mod:`state`. The async ``rag_controller``
coroutines are bridged to synchronous calls via
``utils.do_async_to_sync`` so the Flask handlers and the background
worker can call them directly.
"""

from typing import Any, Dict, List

import logging_config
import model_adapters
import rag_controller
import state
import utils

corpus_logger = logging_config.get_logger("corpus")


def search_corpus(
    query: str,
    top_k: int,
    shard_k: int = 20,
    subsets: List[str] = None,
    rag_algo_choice: int = 5,
) -> Dict[str, Any]:
    """
    Retrieval-only search using rag_controller.search_references(state,...).

    Mirrors the known-good JS:
      - centroid routing to shard_k shards
      - full scoring on those shards
      - returns top_k results
    """
    if not state.ensure_state():
        corpus_logger.error("Search system not initialized. Did boot() work?")
        raise RuntimeError(
            state.get_last_init_error() or "Search system not initialized"
        )

    retrieval_state = state.get_retrieval_state()

    async def _run():
        """Await the reference search and return its ranked results."""
        corpus_logger.info(f"Doing reference search for query: {query[:20]}")
        return await rag_controller.search_references(
            retrieval_state,
            query,
            top_k=int(top_k),
            centroid_k=int(shard_k),
            verbose=False,
            subsets=subsets,
            rag_algo_choice=rag_algo_choice,
        )

    out = utils.do_async_to_sync(_run)() or {}
    if not isinstance(out, dict):
        return {"results": [], "num_results": 0, "query": query}

    # Ensure JSON-safe
    results = out.get("results", [])
    if isinstance(results, list):
        for r in results:
            if isinstance(r, dict):
                # cast numpy scalars if any leaked through
                v = r.get("score_tfidf")
                if v is not None:
                    try:
                        r["score_tfidf"] = float(v)
                    except Exception:
                        pass
                v = r.get("score")
                if v is not None:
                    try:
                        r["score"] = float(v)
                    except Exception:
                        pass

    return out


def prepare_chat_prompt(
    model_type: str,
    query: str,
    *,
    use_rag: bool = True,
    use_double_prompt: bool = False,
    subsets: List[str] = None,
    rag_algo_choice: int = 5,
    prompt_type: int = 0,
    top_k: int = 10,
    context_k: int = 10,
) -> Dict[str, Any]:
    """Build everything needed to start a streamed generation, without
    generating.

    The streaming sibling of the first half of :func:`chat_with_corpus` /
    :func:`rag_controller.ask`: it truncates the question, (when
    ``use_rag``) retrieves references and builds the context block, picks the
    system prompt, and assembles the USER content string — then returns those
    pieces so the caller (the ``/api/chat/stream`` route) can stream
    ``model_adaptor.generate_stream(user_content, system_prompt=...)`` itself
    and forward chunks to the browser. Reference docs are fetched afterward
    via :func:`fetch_chat_references` once the full answer is known.

    The prompt assembly mirrors ``rag_controller.ask`` /
    ``ask_model_only`` exactly so the streamed answer matches what the
    buffered/queued path would have produced for the same inputs.

    Args:
        model_type: Adapter key (must resolve via rag_controller).
        query: The user's question.
        use_rag: When True, retrieve context and include a documents block.
        use_double_prompt: Self-restatement prompting variant.
        subsets: Optional corpus subsets to restrict retrieval to.
        rag_algo_choice: Retrieval algorithm variant selector.
        prompt_type: Index selecting which system prompt to use.
        top_k: Max references to retrieve for context.
        context_k: Max docs fed to context assembly.

    Returns:
        Dict with ``model_adaptor``, ``system_prompt``, ``user_content``,
        ``truncated`` (bool), and ``question`` (the cleaned/truncated query).

    Raises:
        RuntimeError: If the search system isn't initialized or the model
            adapter can't be resolved.
    """
    retrieval_state = state.get_retrieval_state()
    if retrieval_state is None:
        raise RuntimeError("Search system not initialized")

    model_adaptor = rag_controller.get_model_type(model_type)
    if model_adaptor is None:
        raise RuntimeError(
            f"Could not get valid model adapter for type: {model_type}"
        )

    q = (query or "").strip()

    async def _run():
        """Retrieve (optional), build context, and assemble the user content."""
        qq, truncated = rag_controller.truncate_question(q)

        context = ""
        if use_rag:
            refs = await rag_controller.search_references(
                retrieval_state,
                qq,
                top_k=int(top_k),
                verbose=False,
                subsets=subsets,
                rag_algo_choice=rag_algo_choice,
            )
            docs = refs.get("results", []) if isinstance(refs, dict) else []
            context = rag_controller.build_context_v2(
                docs, qq, state=retrieval_state, context_k=int(context_k)
            )

        system_prompt = model_adapters.get_system_prompt(prompt_type)

        # USER content only (system prompt is passed to generate() separately).
        # Mirrors rag_controller.ask / ask_model_only — keep these in sync.
        if use_rag:
            if use_double_prompt:
                question_block = (
                    "Agent Question:\n"
                    f"{qq}\n\n"
                    "Before answering: restate the question in your own words "
                    "and identify what would constitute strong evidence. Then "
                    "answer.\n\n"
                )
            else:
                question_block = "Agent Question:\n" f"{qq}\n\n"
            if str(context).strip():
                docs_block = (
                    "Documents (use if helpful, otherwise ignore):\n"
                    f"{context}"
                )
            else:
                docs_block = ""
            user_content = f"{question_block}{docs_block}"
        else:
            if use_double_prompt:
                user_content = "Agent Question:\n" f"{qq}\n" f"{qq}\n"
            else:
                user_content = "Agent Question:\n" f"{qq}\n"

        return {
            "system_prompt": system_prompt,
            "user_content": user_content,
            "truncated": truncated,
            "question": qq,
        }

    out = utils.do_async_to_sync(_run)()
    out["model_adaptor"] = model_adaptor
    return out


def fetch_chat_references(
    query: str,
    answer: str,
    *,
    subsets: List[str] = None,
    rag_algo_choice: int = 5,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Fetch reference docs for a completed answer (post-generation step).

    The streaming sibling of the reference-retrieval half of
    :func:`chat_with_corpus`: builds the ``question + " " + answer``
    retrieval text, bounds it, and runs ``search_references`` so the
    ``/api/chat/stream`` route can attach citations to its ``done`` event.

    Args:
        query: The original user question.
        answer: The full generated answer.
        subsets: Optional corpus subsets to restrict retrieval to.
        rag_algo_choice: Retrieval algorithm variant selector.
        top_k: Max references to return.

    Returns:
        A list of reference doc dicts (possibly empty).
    """
    retrieval_state = state.get_retrieval_state()
    if retrieval_state is None:
        return []

    q = (query or "").strip()

    async def _run():
        """Build the retrieval text and fetch reference docs."""
        retrieval_text = (q + " " + (answer or "")).strip()
        try:
            retrieval_text = rag_controller.clean_retrieval_text(retrieval_text)
        except Exception:
            pass
        words = retrieval_text.split()
        if len(words) > 900:
            retrieval_text = " ".join(words[:900])

        pack = await rag_controller.search_references(
            retrieval_state,
            retrieval_text,
            top_k=int(top_k),
            verbose=False,
            entity_source_query=q,
            fulltext_query=retrieval_text,
            subsets=subsets,
            rag_algo_choice=rag_algo_choice,
        )
        if isinstance(pack, dict):
            docs = pack.get("results", []) or []
        elif isinstance(pack, list):
            docs = pack
        else:
            docs = []
        if not isinstance(docs, list):
            docs = []
        return docs[: max(0, int(top_k) or 0)]

    return utils.do_async_to_sync(_run)()


def chat_with_corpus(
    model_type: str,
    query: str,
    top_k: int = 10,
    shard_k: int = 20,
    use_rag: bool = True,
    use_double_prompt=False,
    subsets: List[str] = None,
    rag_algo_choice: int = 5,
    prompt_type: int = 0,
) -> Dict[str, Any]:
    """
    Returns (answer: str, docs: list[dict])

    Retrieval behavior:
      - Answer is generated from the *question* (rag_controller.ask does its own retrieval)
      - References are fetched using:
          retrieval_text = question + " " + answer   (chat mode parity goal)
        so the TF-IDF query vector reflects both user intent and the model’s salient terms.
    """
    retrieval_state = state.get_retrieval_state()
    if retrieval_state is None:
        raise RuntimeError("Search system not initialized")

    q = (query or "").strip()
    if not q:
        return ("", [])

    model_adaptor = rag_controller.get_model_type(model_type)
    if model_adaptor is None:
        raise RuntimeError(
            f"Could not get valid model adapter for type: {model_type}"
        )

    async def _run():
        """Await the LLM answer pipeline (retrieval, context build, generation)."""
        # 1) LLM answer (rag_controller.ask does its own retrieval + context building. Skipped if rag toggled off)
        if use_rag:
            answer = await rag_controller.ask(
                model_adaptor,
                retrieval_state,
                q,
                verbose=True,
                use_double_prompt=use_double_prompt,
                subsets=subsets,
                rag_algo_choice=rag_algo_choice,
                prompt_type=prompt_type,
            )
        else:
            answer = await rag_controller.ask_model_only(
                model_adaptor,
                retrieval_state,
                q,
                verbose=False,
                use_double_prompt=use_double_prompt,
                prompt_type=prompt_type,
            )

        answer = str(answer or "").strip()

        # 2) Build retrieval text for references (query + answer)
        if not use_rag:
            return (answer, [])
        retrieval_text = (q + " " + answer).strip()
        try:
            retrieval_text = rag_controller.clean_retrieval_text(retrieval_text)
        except Exception:
            # If you ever rename/move it, don't fail chat
            pass

        # Safety: keep retrieval text bounded (prevents huge TF maps / slow scoring)
        # Tune as needed; ~800–1200 words is plenty for TF-IDF.
        words = retrieval_text.split()
        if len(words) > 900:
            retrieval_text = " ".join(words[:900])

        # extra logging
        corpus_logger.info(
            "chat refs retrieval_text words=%d", len(retrieval_text.split())
        )

        # 3) Fetch reference docs using retrieval_text
        pack = await rag_controller.search_references(
            retrieval_state,
            retrieval_text,
            top_k=int(top_k),
            verbose=False,
            entity_source_query=q,
            fulltext_query=retrieval_text,
            subsets=subsets,
            rag_algo_choice=rag_algo_choice,
        )

        docs = []
        if isinstance(pack, dict):
            results = pack.get("results", []) or []
            docs = results
        elif isinstance(pack, list):
            docs = pack

        if not isinstance(docs, list):
            docs = []

        return (answer, docs[: max(0, int(top_k) or 0)])

    return utils.do_async_to_sync(_run)()

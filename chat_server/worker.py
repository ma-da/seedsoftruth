"""Background job-processing worker.

Runs a daemon thread that drains the ``rag_controller`` job queue: for
each queued chat job it CAS-claims the DB row ('queued' -> 'processing'),
waits for the job's model backend to report ready, runs
``corpus.chat_with_corpus``, stores the reply + references as a JSON blob
on the job row (read back by ``GET /api/job/<id>``) and — when the user
opted in — emails the response. Started once per process by
``init_worker``.
"""

import json
import logging
import threading
import time

import config
import corpus
import db
import email_responses
import logging_config
import rag_controller
import runtime

worker_logger = logging_config.get_logger("worker")

# Due to GIL, booleans should be thread-safe.
app_is_shutdown = False


def _maybe_send_response_email(
    job_id: str, failed: bool, references=None
) -> None:
    """
    After a queued job finishes (success or failure), check whether the
    user attached an email address and dispatch the response via
    email_responses.send_response_email if so.

    Uses db.claim_email_for_send (atomic 'pending → sending') as the
    synchronization primitive so the worker can never double-send with
    the API's race-branch — whichever code path successfully claims
    the row is the one that delivers; the other no-ops.

    Best-effort: any error here is logged but never propagated, so a
    flaky mailer can't take out the worker thread.
    """
    try:
        if not email_responses.is_enabled():
            return

        # Atomically take ownership. If the row isn't 'pending' (no
        # email attached, already sent, or another path beat us to it)
        # this returns False and we drop out cleanly.
        if not db.claim_email_for_send(job_id):
            return

        snapshot = db.get_email_for_job(job_id)
        if not snapshot:
            # Vanishingly unlikely (we just claimed it) but guard anyway.
            db.mark_email_status(job_id, "failed")
            return
        email_addr = snapshot.get("email")
        if not email_addr:
            # Shouldn't happen — claim only succeeds when email_status
            # is 'pending', which is only set by attach_email which
            # also sets email. Defensive cleanup.
            db.mark_email_status(job_id, "failed")
            return

        ok = email_responses.send_response_email(
            email_addr,
            snapshot.get("prompt") or "",
            snapshot.get("response") or "",
            references=references,
            failed=failed,
        )
        db.mark_email_status(job_id, "sent" if ok else "failed")
    except Exception:
        worker_logger.exception(
            "Email-response dispatch failed for job_id=%s (best-effort, swallowed)",
            job_id,
        )
        # If we claimed but then crashed mid-send, the row would be
        # stuck in 'sending'. Best-effort flip to 'failed' so an
        # operator / sweeper can spot it.
        try:
            db.mark_email_status(job_id, "failed")
        except Exception:
            pass


def worker_body() -> None:
    """Run the background job-processing loop until the app shuts down.

    Repeatedly peeks at the head of the rag_controller job queue and,
    once the job's model backend reports ready, CAS-claims the DB row
    ('queued' -> 'processing') and runs ``chat_with_corpus`` for it. The
    reply and references are stored as a JSON blob on the job row (read
    back by ``GET /api/job/<id>``) and, when opted in, emailed to the
    user. Cold backends are nudged awake via warmup requests while
    waiting. A per-iteration latency line is logged for diagnostics.

    The loop body is wrapped in an outer try/except so that an unexpected
    exception cannot silently kill this daemon thread and stall the queue.

    Intended to run on a daemon thread started by ``init_worker``.
    """
    global app_is_shutdown
    MODEL_NOT_READY_REPORTING_INTERVAL = 5
    WORKER_NO_WORK_SLEEP_INTERVAL_SECS = 5
    WORKER_NO_QUEUED_JOB_REPORTING_INTERVAL = 20

    model_not_ready_interval = (
        MODEL_NOT_READY_REPORTING_INTERVAL - 1
    )  # let's first report happen sooner
    worker_no_queued_job_interval = WORKER_NO_QUEUED_JOB_REPORTING_INTERVAL - 1

    worker_ready_report = False
    worker_logger.info("App worker started")

    while not app_is_shutdown:
        # OUTER guard: the worker is a daemon thread; if an exception
        # ever escapes the loop body, the thread dies silently and the
        # queue grinds to a halt with no alarm. We sleep+continue so the
        # next iteration tries again. The cost of a tight crash loop is
        # bounded by the heartbeat sleep below.
        try:
            # Peek at the head of the queue (non-destructive). We need the
            # job's model_type to gate readiness on the correct backend
            # rather than the global default.
            queued_job = rag_controller.get_next_queued_job()
            if not queued_job:
                worker_no_queued_job_interval = (
                    worker_no_queued_job_interval + 1
                )
                if (
                    worker_no_queued_job_interval
                    > WORKER_NO_QUEUED_JOB_REPORTING_INTERVAL
                ):
                    worker_logger.info(
                        "Worker thread heartbeat. No job work to do."
                    )
                    worker_no_queued_job_interval = 0

                time.sleep(WORKER_NO_WORK_SLEEP_INTERVAL_SECS)
                continue
            else:
                worker_no_queued_job_interval = 0

            worker_logger.info(
                f"Background worker making health request for model_type {queued_job.model_type}"
            )
            model_ready = rag_controller.is_model_type_ready(
                queued_job.model_type
            )
            if not model_ready:
                model_not_ready_interval = model_not_ready_interval + 1
                worker_ready_report = False

                if (
                    model_not_ready_interval
                    > MODEL_NOT_READY_REPORTING_INTERVAL
                ):
                    worker_logger.info(
                        f"Model {queued_job.model_type} still not ready. Worker body waiting."
                    )
                    model_not_ready_interval = 0

                # Nudge the right backend awake while we wait.
                rag_controller.send_warmup_for(queued_job.model_type)
                time.sleep(WORKER_NO_WORK_SLEEP_INTERVAL_SECS)
                continue
            else:
                model_not_ready_interval = 0

            if not worker_ready_report:
                worker_logger.info(
                    f"Model {queued_job.model_type} is ready. Worker is active."
                )
                worker_ready_report = True

            # Atomically flip status 'queued' -> 'processing'. The CAS is
            # `WHERE id=? AND status='queued'`, so it can either:
            #   (a) succeed (cas_ok=True) — we own it, proceed
            #   (b) return False — row missing, marked failed by sweep, or
            #       (hypothetically under workers>1) claimed by another
            #       worker. Either way the row is in a state where we
            #       must NOT call chat_with_corpus, so pop and skip.
            #   (c) raise — transient DB hiccup. Do NOT pop, so the next
            #       iteration retries. Sleep briefly to avoid a hot loop.
            try:
                cas_ok = db.mark_processing(queued_job.job_id)
            except Exception:
                worker_logger.exception(
                    f"worker_body: mark_processing raised for job_id {queued_job.job_id}; "
                    f"leaving in queue for retry"
                )
                time.sleep(WORKER_NO_WORK_SLEEP_INTERVAL_SECS)
                continue

            if not cas_ok:
                worker_logger.warning(
                    f"worker_body: mark_processing CAS missed for job_id {queued_job.job_id} "
                    f"(row missing, already terminal, or claimed by another worker); skipping"
                )
                rag_controller.pop_next_queued_job()
                continue

            # Track in-flight chat work for the /api/queue depth widget. Lives
            # inside try/except so a counter blip can't ever lose a decrement.
            runtime.inflight_chat_reqs.inc()

            # Latency breakdown:
            #   queue_wait = time the job sat in the in-mem queue before the
            #                worker picked it up + cleared readiness checks
            #   processing = time chat_with_corpus actually spent on the model
            #   total      = queue_wait + processing
            # `queued_at` is set in rag_controller.queue_job at append time;
            # 0.0 means the field is missing (defensive fallback for any code
            # path that constructs a QueuedJob directly without going through
            # queue_job).
            queued_at = getattr(queued_job, "queued_at", 0.0) or 0.0
            proc_start_ts = time.time()
            queue_wait_secs = (
                (proc_start_ts - queued_at) if queued_at else 0.0
            )
            job_outcome = "unknown"
            try:
                worker_logger.info(
                    f"Worker chat_with_corpus for model_type {queued_job.model_type} job_id {queued_job.job_id} with user_id {queued_job.user_id}..."
                )
                answer, docs = corpus.chat_with_corpus(
                    queued_job.model_type,
                    queued_job.prompt,
                    top_k=10,
                    use_rag=getattr(queued_job, "use_rag", True),
                    use_double_prompt=config.USE_DOUBLE_PROMPT,
                    subsets=queued_job.subsets,
                    rag_algo_choice=queued_job.rag_algo_choice,
                    prompt_type=getattr(queued_job, "prompt_type", 0),
                )

                # clean up docs of copyrighted material
                cleaned_docs = rag_controller.clean_rag_references(docs)

                # Store reply + references as a JSON blob in the existing
                # `response` TEXT column. /api/job/<id> parses this back out;
                # legacy plain-string rows are read-compatible (see
                # api_job_status). This is what eliminates the references-lost
                # bug on the formerly-queued path.
                payload_json = json.dumps({
                    "reply": answer,
                    "references": cleaned_docs,
                })

                worker_logger.info(
                    f"Worker query job_id {queued_job.job_id} was executed succesfully. Marking done in db."
                )
                db.mark_done(queued_job.job_id, payload_json)
                job_outcome = "done"

                # If the user opted into email delivery on the queue response,
                # send it now. No-op when the feature flag is off or no email
                # was attached.
                _maybe_send_response_email(
                    queued_job.job_id,
                    failed=False,
                    references=cleaned_docs,
                )

            except RuntimeError as e:
                worker_logger.error(
                    f"chat_with_corpus failed in worker with runtime error: {e}"
                )
                db.mark_failed(
                    queued_job.job_id, f"Search system not initialized: {e}"
                )
                job_outcome = "failed_runtime"
                _maybe_send_response_email(queued_job.job_id, failed=True)

            except Exception as e:
                worker_logger.error(
                    f"chat_with_corpus failed in worker with exception: {e}"
                )
                db.mark_failed(queued_job.job_id, f"Chat failed: {e}")
                job_outcome = "failed_exception"
                _maybe_send_response_email(queued_job.job_id, failed=True)

            finally:
                # Latency summary, emitted exactly once per terminal job (done
                # or failed), so any single grep on `LATENCY job_id=` reveals
                # the queue_wait / processing / total breakdown for diagnostic
                # purposes (perf regressions, slow models, queue backlog).
                proc_end_ts = time.time()
                processing_secs = proc_end_ts - proc_start_ts
                total_secs = (
                    (proc_end_ts - queued_at) if queued_at else processing_secs
                )
                worker_logger.info(
                    "LATENCY job_id=%s user_id=%s model_type=%s outcome=%s "
                    "queue_wait=%.3fs processing=%.3fs total=%.3fs",
                    queued_job.job_id,
                    queued_job.user_id,
                    queued_job.model_type,
                    job_outcome,
                    queue_wait_secs,
                    processing_secs,
                    total_secs,
                )
                runtime.inflight_chat_reqs.dec()

            rag_controller.pop_next_queued_job()

        except Exception:
            # Outer guard: anything raised by get_next_queued_job, the
            # readiness check, or any setup path above. Log loudly and
            # sleep briefly so the next iteration retries instead of the
            # thread dying. Bounded by WORKER_NO_WORK_SLEEP_INTERVAL_SECS.
            worker_logger.exception(
                "worker_body: unhandled exception in main loop; sleeping then continuing"
            )
            try:
                time.sleep(WORKER_NO_WORK_SLEEP_INTERVAL_SECS)
            except Exception:
                pass

    worker_logger.info("App worker shutdown")


def init_worker() -> None:
    """Start the background job worker on a daemon thread."""
    worker = threading.Thread(target=worker_body)
    worker.daemon = True
    worker.start()
    logging.info("Started daemon worker thread")

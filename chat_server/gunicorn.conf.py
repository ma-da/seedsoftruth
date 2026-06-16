# gunicorn.conf.py
#
# The web-server modules live under chat_server/ but are imported flat
# (e.g. `import config`). We launch from the repo root so cwd-relative
# paths (like db/app.db) keep resolving, and add chat_server/ to the
# import path here so the flat imports — and the WSGI target `app:app` —
# resolve. Run with: gunicorn -c chat_server/gunicorn.conf.py app:app
pythonpath = "chat_server"

bind = "127.0.0.1:8000"
workers = 1

# Use threaded workers so concurrent /api/job/<id> polls + /api/status
# health checks don't contend on a single sync worker. With sync workers,
# one in-flight request blocks every other route, which made the UI's
# periodic health poll stall during the legacy synchronous chat path.
#
# We stay at workers=1 on purpose: the queue, rate limiter, and other
# in-process state in rag_controller live in one Python process and
# are not safe to split across multiple worker processes. Threads in
# the same worker share that state and are protected by the existing
# locks (see _state_lock in app.py and the locks inside the queue
# helpers in rag_controller). See SECURITY.md for the workers=1
# rationale.
worker_class = "gthread"
threads = 8

# /api/chat is now async — no request handler blocks on the LLM. The
# longest legitimate handler is a single DB write, so a tight timeout
# guards against a future regression that re-introduces a long sync
# call without anyone noticing. The background worker thread is NOT
# bound by this timeout — gunicorn only times out HTTP request
# handlers, not threads spawned by the app.
timeout = 30
graceful_timeout = 30

# Optional but useful
accesslog = "-"
errorlog = "-"
loglevel = "info"

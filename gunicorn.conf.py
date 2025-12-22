# gunicorn.conf.py
bind = "127.0.0.1:8000"
workers = 2

# IMPORTANT: allow long HF calls
timeout = 180
graceful_timeout = 30

# Optional but useful
accesslog = "-"
errorlog = "-"
loglevel = "info"

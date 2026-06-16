#!/bin/bash
# Run from the repo root: `bash scripts/run.sh`
# The web-server modules live under chat_server/ but import each other
# flat (e.g. `import config`), so put chat_server/ on the import path.
# cwd stays the repo root so cwd-relative paths (db/app.db) resolve.
source venv311/bin/activate
export FLASK_APP=app
export PYTHONPATH="chat_server${PYTHONPATH:+:$PYTHONPATH}"
flask run

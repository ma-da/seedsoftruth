#!/bin/bash
# Run from the repo root: `bash scripts/run_debug.sh`
# chat_server/ holds the flat-imported web-server modules; put it on the
# path. cwd stays the repo root so db/app.db etc. resolve.
source venv311/bin/activate
export FLASK_APP=app
export PYTHONPATH="chat_server${PYTHONPATH:+:$PYTHONPATH}"
export FLASK_ENV=development   # enables debug
flask run --debug

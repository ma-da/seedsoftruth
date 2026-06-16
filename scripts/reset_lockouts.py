#!/usr/bin/env python3
"""
Reset IP and/or session lockouts on a running Seeds of Truth server.

The admin endpoint (``/api/admin/clear-lockouts``) accepts two auth
paths:

  - Direct local requests: anything reaching gunicorn from a loopback
    peer without proxy headers is allowed without a token. That covers
    the common case of SSHing onto the server and running this script.

  - Remote requests: must carry ``SOT_ADMIN_TOKEN``. Used when you
    want to clear lockouts from outside the server (over HTTPS).

So by default this script runs without a token — it just assumes
you're on the server. If you're invoking it remotely, set
``SOT_ADMIN_TOKEN`` in your shell (matching the value the server was
started with) or pass ``--token``.

Typical local usage:

    python scripts/reset_lockouts.py --ip 127.0.0.1     # clear one IP
    python scripts/reset_lockouts.py --ip '*'           # clear all

Remote usage:

    SOT_URL=https://seedsoftruth.example.org \\
    SOT_ADMIN_TOKEN=<token> \\
        python scripts/reset_lockouts.py --ip 1.2.3.4

Note: IP-level state lives in the running worker's memory. The
per-session lockout lives in the *caller's* cookie, so you can only
clear it for yourself via ``--session`` — typically easier to just
open a new browser session / clear cookies.

Stdlib-only; no requirements to install.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    """Parse args and POST to /api/admin/clear-lockouts to clear lockouts.

    Builds the request payload from ``--ip``/``--session``/``--token``, sends
    it to the admin endpoint, and prints the (JSON) response. Surfaces helpful
    hints on 403 responses.

    Returns:
        Process exit code: 0 on success, 1 on HTTP/connection error, 2 if
        neither ``--ip`` nor ``--session`` was provided.
    """
    p = argparse.ArgumentParser(
        description="Clear IP/session lockouts via /api/admin/clear-lockouts.",
    )
    p.add_argument(
        "--url",
        default=os.environ.get("SOT_URL", "http://localhost:5000"),
        help="Base URL of the running app (default: $SOT_URL or http://localhost:5000)",
    )
    p.add_argument(
        "--token",
        default=os.environ.get("SOT_ADMIN_TOKEN", ""),
        help="Admin token. Only required for REMOTE invocations; local "
             "requests on the server are accepted without a token. "
             "(default: $SOT_ADMIN_TOKEN)",
    )
    p.add_argument(
        "--ip",
        default="",
        help="IP to clear, or '*' to clear all IPs.",
    )
    p.add_argument(
        "--session",
        action="store_true",
        help="Also clear THIS request's session lockout. "
             "(Usually not what you want; restart your browser instead.)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds (default: 10).",
    )
    args = p.parse_args()

    if not args.ip and not args.session:
        print(
            "error: nothing to clear. Pass --ip <addr|*> and/or --session.",
            file=sys.stderr,
        )
        return 2

    payload: dict = {}
    if args.token:
        payload["token"] = args.token
    if args.ip:
        payload["ip"] = args.ip
    if args.session:
        payload["session"] = True

    url = args.url.rstrip("/") + "/api/admin/clear-lockouts"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            body = r.read().decode("utf-8", "replace")
            try:
                parsed = json.loads(body)
                print(json.dumps(parsed, indent=2))
            except json.JSONDecodeError:
                print(body)
        return 0
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        print(f"HTTP {e.code}: {body}", file=sys.stderr)
        if e.code == 403:
            if not args.token:
                print(
                    "hint: 403 from a local invocation usually means this "
                    "request went through a proxy (you're not on the server, "
                    "or the URL points at the public hostname). Set "
                    "SOT_ADMIN_TOKEN or pass --token for remote use.",
                    file=sys.stderr,
                )
            else:
                print(
                    "hint: bad token? Check it matches the value Flask was "
                    "started with.",
                    file=sys.stderr,
                )
        return 1
    except urllib.error.URLError as e:
        print(f"connection error: {e.reason}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

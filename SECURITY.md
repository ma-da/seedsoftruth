# Security Policy

## Scope and status

Seeds of Truth is **alpha software**. The reference deployment is a private invite-only service; the codebase has not been hardened for the open internet. There are known security gaps tracked in [`ALPHA_BLOCKERS.md`](ALPHA_BLOCKERS.md) that an operator must close before exposing this to untrusted users. Read that file before deploying.

This policy covers the published source code in this repository. Issues in third-party dependencies (Flask, gunicorn, spaCy, etc.) should be reported to those projects directly.

## Supported versions

| Version       | Status     | Receives security fixes |
|---------------|------------|-------------------------|
| `main` branch | Active     | Yes                     |
| Tagged releases | None yet — the project has not yet cut a v1.0 |                         |

Once v1.0 is tagged, this table will be updated to specify which release lines receive backported fixes.

## Reporting a vulnerability

**Please do not file public GitHub issues for security vulnerabilities.**

Email the maintainers privately at:

> **`<TODO: set a security contact address — e.g. security@your-domain.tld or a personal address you monitor>`**

Include:

- A description of the issue and its impact
- Steps to reproduce, or a proof-of-concept
- The version (commit hash) you are testing against
- Whether you are willing to be credited in the fix announcement

You should expect:

- An acknowledgement within **3 working days**
- An initial assessment within **7 working days**
- A fix or mitigation plan within **30 days** for high-severity issues; longer for issues that require larger refactors, in which case we will keep you posted

If you do not hear back in those windows, follow up — your message may have been missed.

## Disclosure policy

We follow a **coordinated-disclosure** model:

1. You report the issue privately.
2. We confirm and develop a fix.
3. We agree on a disclosure date with you (typically the day a fix lands in `main`, or up to 90 days after the report if the fix is invasive).
4. We publish a security advisory and credit the reporter unless they prefer otherwise.

If you discover a vulnerability that is already being actively exploited, we will work with you on an accelerated timeline.

## What we consider in-scope

Issues we want to know about:

- Authentication or session bypasses (the password-gate, session-cookie forging, etc.)
- Injection attacks (SQL, prompt injection escalations that bypass the system prompt, command injection)
- Information disclosure (logs leaking secrets, tracebacks leaking internal paths, side-channel leaks)
- Denial-of-service vectors against the LLM call path or the queue
- Cross-site request forgery, cross-site scripting, open redirects
- Broken access control on any `/api/*` endpoint
- Supply-chain issues in `requirements.txt` or `package.json`

## What we consider out-of-scope

- **Reports of "no rate limit on `/api/unlock`," "no CSRF on POST endpoints," or similar items already documented in [`ALPHA_BLOCKERS.md`](ALPHA_BLOCKERS.md).** We know. They are tracked. PRs welcome.
- Issues that require physical access to the operator's machine.
- Vulnerabilities in unsupported third-party services (e.g., a HuggingFace endpoint compromise).
- Theoretical timing attacks against `hmac.compare_digest`-protected comparisons that do not have a practical impact.
- Best-practice recommendations without a concrete vulnerability ("you should add Subresource Integrity").

## Operator security checklist

If you are deploying this software, please at minimum:

1. **Set `FLASK_SECRET_KEY` to a long random hex** — never rely on the in-code default.
2. **Set `SOT_PASSWORDS` to a strong, rotated value** — this is currently a shared password gate; everyone with the password has the same access level.
3. **Front the app with a reverse proxy (nginx, Caddy, Cloudflare)** — handle TLS, rate limiting, and IP-level blocking there. Do not expose gunicorn directly.
4. **Lower `HF_TIMEOUT_SECS` and `HF_MAX_ATTEMPTS`** from the in-code defaults (900 / 10) to something defensible (~120 / 3 in production).
5. **Restrict `/api/unlock` at the proxy layer** with per-IP rate limiting until application-level rate limiting lands.
6. **Scrub logs before sharing** — the current code logs the unlock password in plaintext. Until that is fixed in the app, redact `pw=` lines before any log handoff.
7. **Run with `workers = 1`** in `gunicorn.conf.py` until the queue is moved out of the Flask process. Multiple workers will silently lose jobs.
8. **Monitor `/api/status`** to catch silent worker-thread death.
9. **Keep your corpus database read-only at the OS level** — the retrieval code only needs read access; an attacker who escalates to write would be able to inject documents.
10. **Subscribe to repository security advisories** on GitHub.

## Acknowledgments

We will list reporters who have responsibly disclosed vulnerabilities here, with their permission, after fixes have shipped.

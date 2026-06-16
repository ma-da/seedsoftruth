"""
Email-response feature for queued chat completions.

When a chat request is queued (HTTP 503 / model-not-ready), the user can
opt to be emailed when the response is ready instead of waiting in-tab.
This module owns the feature flag, address validation, and the actual
mail-send call.

Feature is gated by `is_enabled()`, which requires BOTH:
  - EMAIL_RESPONSES_ENABLED=1 in env
  - sufficient SMTP config to actually send (host, from address)

So the flag can never accidentally turn on in a state where it would try
to send and fail. While SMTP is being procured the send function logs
the would-be email instead of dispatching it; swap that one function
when SMTP is ready and flip the env var.
"""

from __future__ import annotations

import logging
import os
import re
from typing import List, Optional

logger = logging.getLogger("seedsoftruth.email")


# ------------------ Config ------------------


def _env(name: str, default: str = "") -> str:
    """Read an environment variable, trimmed, with a fallback.

    Args:
        name: Environment variable to read.
        default: Value to use when the variable is unset or empty.

    Returns:
        The whitespace-stripped value, or ``default`` if unset/empty.
    """
    return (os.environ.get(name) or default).strip()


# Read once at import time. Module-level so the values appear in the same
# place as other env-config in the codebase (rag_controller, model_adapters).
EMAIL_RESPONSES_ENABLED_RAW = _env("SOT_EMAIL_RESPONSES_ENABLED", "1")
SMTP_HOST = _env("SOT_SMTP_HOST")
SMTP_PORT = int(_env("SOT_SMTP_PORT", "587") or "587")
SMTP_USER = _env("SOT_SMTP_USER")
SMTP_PASSWORD = os.environ.get("SOT_SMTP_PASSWORD", "")  # don't strip secrets
EMAIL_FROM_ADDRESS = _env("SOT_EMAIL_FROM_ADDRESS")
EMAIL_REPLY_TO = _env("SOT_EMAIL_REPLY_TO")
EMAIL_FROM_NAME = _env("SOT_EMAIL_FROM_NAME", "Seeds of Truth")

# Public-facing app URL used in email bodies so users can click back. Falls
# back to a reasonable default; production deploys should set this.
APP_PUBLIC_URL = _env("APP_PUBLIC_URL", "https://seedsoftruth.peerservice.org")


def is_enabled() -> bool:
    """
    Returns True only if the feature flag is on AND SMTP is configured
    well enough to send. The two-check pattern means we can't ship a state
    where the UI offers email but the server can't actually deliver.
    """
    flag_on = EMAIL_RESPONSES_ENABLED_RAW.lower() in ("1", "true", "yes", "on")
    if not flag_on:
        return False

    if not SMTP_HOST or not EMAIL_FROM_ADDRESS:
        # Flag set but SMTP not configured. Log loudly the first time we
        # check so an operator notices the misconfiguration.
        if not _is_enabled.warned:
            logger.warning(
                "EMAIL_RESPONSES_ENABLED is set but SMTP config is incomplete "
                "(missing SMTP_HOST or EMAIL_FROM_ADDRESS). Email-response "
                "feature will remain disabled."
            )
            _is_enabled.warned = True
        return False

    return True


# Attribute-on-function trick to one-shot the warning without a global.
def _is_enabled():  # pragma: no cover - sentinel only
    """Sentinel function used only as a home for the ``warned`` attribute.

    The attribute-on-function trick lets :func:`is_enabled` emit its
    "SMTP misconfigured" warning exactly once without introducing a
    module-level global.
    """
    pass


_is_enabled.warned = False


# ------------------ Validation ------------------

# Pragmatic email regex. Not RFC-perfect (nothing reasonably is), but
# rejects the obvious junk (no @, spaces, missing TLD, leading/trailing
# dots). Real validation happens at SMTP-send time; this is just to keep
# typos from polluting the jobs table.
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")

MAX_EMAIL_LEN = 254  # RFC 5321 practical max


def validate_email(addr: str) -> Optional[str]:
    """
    Returns a cleaned, lowercased email if valid, or None if not.
    """
    if not isinstance(addr, str):
        return None
    cleaned = addr.strip()
    if not cleaned or len(cleaned) > MAX_EMAIL_LEN:
        return None
    if not _EMAIL_RE.match(cleaned):
        return None
    return cleaned.lower()


# ------------------ Send ------------------


def _format_subject(prompt: str) -> str:
    """Build the email subject line from the user's prompt.

    Uses the first line of ``prompt``, truncated to 60 characters, as a
    snippet. Falls back to a generic subject when the prompt is empty.

    Args:
        prompt: The original user question.

    Returns:
        The subject line for the response email.
    """
    snippet = (prompt or "").strip().splitlines()[0] if prompt else ""
    snippet = snippet[:60].strip()
    if snippet:
        return f"Seeds of Truth response: {snippet}"
    return "Seeds of Truth response"


def _format_body_text(
    prompt: str,
    response: str,
    references: Optional[List[dict]] = None,
    failed: bool = False,
) -> str:
    """Render the plain-text body of a response email.

    Args:
        prompt: The original user question, echoed back in the body.
        response: The generated answer text.
        references: Optional source citations to append to the body.
        failed: When True, render an apologetic "couldn't process this
            time" message instead of the answer.

    Returns:
        The fully formatted plain-text email body.
    """
    lines: List[str] = []
    if failed:
        lines.append(
            "We weren't able to process your question this time. "
            "Please try again at the link below — you don't need to do "
            "anything else right now."
        )
        lines.append("")
    else:
        lines.append("Here's the response to your question.")
        lines.append("")

    lines.append("Your question:")
    lines.append(prompt or "(no question text on file)")
    lines.append("")

    if not failed:
        lines.append("Response:")
        lines.append(response or "(no response on file)")
        lines.append("")

        if references:
            lines.append("References:")
            for i, ref in enumerate(references[:10], start=1):
                title = (
                    ref.get("title") or ref.get("source_title") or "Reference"
                )
                url = ref.get("source_url") or ref.get("url") or ""
                if url:
                    lines.append(f"  {i}. {title} — {url}")
                else:
                    lines.append(f"  {i}. {title}")
            lines.append("")

    lines.append(f"Continue at {APP_PUBLIC_URL}")
    lines.append("")
    lines.append("— Seeds of Truth")

    return "\n".join(lines)


def send_response_email(
    to_email: str,
    prompt: str,
    response: str,
    references: Optional[List[dict]] = None,
    failed: bool = False,
) -> bool:
    """
    Sends the chat response (or a "we couldn't process this" notice) to
    `to_email`. Returns True on success, False on any error.

    PHASE 1 STUB: while SMTP is being procured this function only logs
    what would have been sent. Phase 2 swaps the body of this function
    for an smtplib / provider-SDK call. Callers don't need to change.
    """
    if not is_enabled():
        logger.info(
            "send_response_email called while feature disabled; dropping "
            "(to=%s, failed=%s)",
            to_email,
            failed,
        )
        return False

    cleaned = validate_email(to_email)
    if not cleaned:
        logger.warning("send_response_email: invalid address %r", to_email)
        return False

    subject = _format_subject(prompt)
    body = _format_body_text(prompt, response, references, failed=failed)

    # ---- PHASE 1 STUB ----
    # Log a clearly-tagged record of what we would have sent. Body is
    # truncated in the log to keep things readable.
    body_preview = body if len(body) <= 600 else body[:600] + " …[truncated]"
    logger.info(
        "[email-stub] would send | to=%s | from=%s | subject=%s\n%s",
        cleaned,
        EMAIL_FROM_ADDRESS,
        subject,
        body_preview,
    )
    # Pretend success in stub mode — the worker should mark the job's
    # email_status='sent' so it doesn't re-attempt forever.
    return True

    # ---- PHASE 2 (uncomment + delete the stub return above when SMTP
    # ---- is wired up):
    #
    # import smtplib
    # from email.message import EmailMessage
    #
    # try:
    #     msg = EmailMessage()
    #     msg["Subject"] = subject
    #     msg["From"] = f"{EMAIL_FROM_NAME} <{EMAIL_FROM_ADDRESS}>"
    #     msg["To"] = cleaned
    #     if EMAIL_REPLY_TO:
    #         msg["Reply-To"] = EMAIL_REPLY_TO
    #     msg.set_content(body)
    #
    #     with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as s:
    #         s.starttls()
    #         if SMTP_USER:
    #             s.login(SMTP_USER, SMTP_PASSWORD)
    #         s.send_message(msg)
    #     logger.info("Sent response email to %s", cleaned)
    #     return True
    # except Exception:
    #     logger.exception("Failed to send response email to %s", cleaned)
    #     return False

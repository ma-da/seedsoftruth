// app.js — Seeds of Truth UI logic (Flask routes + password gate)

/**
 * @fileoverview Frontend single-page-app logic for the Seeds of Truth RAG web
 * app. This vanilla-JavaScript module wires up the entire UI: DOM element
 * caching, light/dark theme and sidebar behavior, the tools popup and its
 * persisted state, custom alert/confirm modals, the chat/search/A-B submission
 * flow against the Flask API, reference card rendering, a small safe
 * markdown-to-HTML renderer, endpoint status and queue polling, the
 * email-response flow for queued requests, and localStorage persistence for
 * tool settings, saved conversations, feedback, and pending emails.
 */

'use strict';

/* =========================================================
   0) TUNABLES / CONSTANTS
   ========================================================= */
const CFG = {
  // Version Info,
  VERSION_NUM: '1.0',
  VERSION_NAME: 'Dingo',

  // LocalStorage keys
  LS_THEME: 'sot-theme',
  LS_SIDEBAR_COLLAPSED: 'sot-sidebar-collapsed',
  LS_TOOLS: 'sot-tools',
  LS_SAVED_CONVOS: 'sot-saved-conversations-v1',
  LS_FEEDBACK: 'sot-feedback-v1',
  LS_PENDING_EMAILS: 'sot-pending-emails-v1',
  LS_LAST_EMAIL: 'sot-last-email-v1', // remembered for autofill convenience

  // How long a pending-email entry survives in localStorage before
  // we assume it's stale (worker died, refresh, etc.).
  PENDING_EMAIL_TTL_MS: 60 * 60 * 1000, // 1 hour

  // When the client's last-known queue depth is at or above this
  // threshold AND the server has reported email_offer:true, opening
  // a new chat request prompts the user to opt into email up-front
  // (before we even send the chat). Set to 0 to disable; raise to
  // make the proactive prompt less aggressive.
  PROACTIVE_EMAIL_THRESHOLD: 3,

  // Limits
  MAX_CONTEXT_TURNS: 5,
  MAX_CONVO_TURNS: 50,
  MAX_SAVED_CONVOS: 25,
  MAX_FEEDBACK_ITEMS: 200,
  MAX_REFS: 10,

  // UI behavior
  DEFAULT_THEME: 'light', // 'light' | 'dark'
  DEFAULT_HISTORY_TURNS: 3, // 0..5
  DEFAULT_MODE: 'chat', // 'search' | 'chat' | 'ab'
  TEXTAREA_MAX_HEIGHT: 140, // px

  // Polling
  STATUS_POLL_MS: 15000,
  QUEUE_POLL_MS: 15000,

  // Per-job polling (GET /api/job/<id>)
  JOB_POLL_INITIAL_MS: 1500,
  JOB_POLL_MAX_MS: 5000,
  JOB_POLL_TOTAL_MS: 10 * 60 * 1000,   // 10 min hard cap
  JOB_POLL_NET_FAIL_NOTIFY: 3,         // surface "lost connection" after N misses

  // SessionStorage resume — recover in-flight job on page reload
  ACTIVE_JOB_SS_KEY: 'sot.activeJob',
  ACTIVE_JOB_RESUME_MAX_AGE_MS: 15 * 60 * 1000,  // 15 min

  // Flask endpoints
  API: {
    UNLOCK: '/api/unlock',
    ACCESS: '/api/access',
    SEARCH: '/api/search',
    CHAT: '/api/chat',
    CHAT_STREAM: '/api/chat/stream',  // POST -> Server-Sent Events
    AB: '/api/ab',
    FEEDBACK: '/api/feedback',
    STATUS: '/api/status',
    QUEUE: '/api/queue',
    JOB: '/api/job',                 // GET /api/job/<id>?user_id=…
    PING: '/api/ping',
    EMAIL_RESPONSE: '/api/email_response',
    EMAIL_RESPONSE_CANCEL: '/api/email_response/cancel',
  },

  // Error codes
  JOB_ID_NONE: 'none',
  DEFAULT_RAG_ALGO_TYPE: 5,
  DEFAULT_PROMPT_TYPE: 1,

  // Developer mode
  // Should only be set to true locally, check-in as false
  DEV_MODE: true,
};

let requestInFlight = false;

// Map of job_id -> { placeholder, startedAt } for jobs the UI is currently
// polling. Multiple concurrent jobs are supported so the user can fire off
// a second question without the first dropping its placeholder bubble.
const activeJobs = new Map();

// Back-compat shim: code paths that just want "is anything in flight" can
// still read activeJobId — it's true when at least one job is being polled.
// Use activeJobs.size for accurate counting.
let activeJobId = null;
function setActiveJobIdHint() {
  // Pick any job_id from the map to satisfy legacy boolean checks (the
  // exact value doesn't matter — the polling-cadence check at the bottom
  // of refreshQueueOnce uses it as a truthy hint, not a key lookup).
  activeJobId = activeJobs.size > 0 ? activeJobs.keys().next().value : null;
}

// Primary chat backend, and the backend we fall back to when it's down.
const PRIMARY_MODEL_TYPE = "spark";
const FALLBACK_MODEL_TYPE = "deepinfra";

// Cached readiness of PRIMARY_MODEL_TYPE specifically. Only ever written from
// a status check that explicitly targeted PRIMARY_MODEL_TYPE — see
// chooseModelTypeForSubmit() and refreshStatusOnce().
let lastKnownSparkReady = null;
let lastSparkHealthCheckAt = 0;

const SPARK_HEALTH_MAX_AGE_MS = 60_000;

const NOT_READY_MSG =
  'Model not ready. We will process your request when it comes online. Please wait for response.';

const SUBSET_COMBOS = [
  {
    combo_name: 'WantToKnow',
    subsets: ['WTK Archive', 'WantToKnow.info', 'PEERS Substack'],
  },
  {
    combo_name: 'Deep Politics',
    subsets: [
      'WTK Archive',
      'WantToKnow.info',
      'PEERS Substack',
      'theblackvault.com',
    ],
  },
  {
    combo_name: 'Health',
    subsets: [
      'WTK Archive',
      'WantToKnow.info',
      'PEERS Substack',
      'childrenshealthdefense.org',
      'usrtk.org',
      'vaccinepapers.org',
      'howdovaccinescauseautism.org',
    ],
  },
  {
    combo_name: 'UFO',
    subsets: [
      'WTK Archive',
      'WantToKnow.info',
      'PEERS Substack',
      'theblackvault.com',
      'newparadigminstitute.org',
    ],
  },
  {
    combo_name: 'Everything',
    subsets: [
      'WTK Archive',
      'WantToKnow.info',
      'PEERS Substack',
      'arlingtoninstitute.org',
      'childrenshealthdefense.org',
      'doortofreedom.org',
      'howdovaccinescauseautism.org',
      'judicialwatch.org',
      'learntherisk.org',
      'newparadigminstitute.org',
      'nypost.com',
      'organicconsumers.org',
      'substack',
      'theblackvault.com',
      'thepulse.one',
      'trialsitenews.com',
      'usrtk.org',
      'vaccinepapers.org',
      'wingmakers.com',
    ],
  },
];

/* =========================================================
   1) DOM LOOKUPS (set in init)
   ========================================================= */
const els = {}; // populated in initDom()

/* =========================================================
   2) STATE
   ========================================================= */
const toolState = {
  historyTurns: CFG.DEFAULT_HISTORY_TURNS,
  mode: CFG.DEFAULT_MODE,
  useRag: true,
  searchGroup: 'Deep Politics',
  modelType: 'spark',
  // Experimental routing/prompt controls, persisted with other tool state
  ragAlgoType: 1,
  promptType: 1,
  // Dev-only: force /api/chat to take the queued path regardless of model
  // readiness. Hidden + ignored outside DEV_MODE.
  forceQueue: false,
};

// client-side turns: { user: string, assistant: string }
const convoTurns = [];
let botMsgCounter = 0;

// feedback modal state
let feedbackTarget = null;

// lock gate
let isUnlocked = false;

/* =========================================================
   3) UTILITIES
   ========================================================= */
/**
 * Parses a value as a base-10 integer and clamps it into [min, max].
 * @param {(number|string)} n Value to parse.
 * @param {number} min Lower bound (inclusive).
 * @param {number} max Upper bound (inclusive).
 * @param {number} fallback Value returned when n is not a valid integer.
 * @returns {number} The clamped integer, or fallback when parsing fails.
 */
function clampInt(n, min, max, fallback) {
  const x = parseInt(n, 10);
  if (Number.isNaN(x)) return fallback;
  return Math.max(min, Math.min(max, x));
}

/**
 * Parses a value as a base-10 integer and clamps it into [1, 10].
 * @param {(number|string)} n Value to parse.
 * @returns {?number} The clamped integer, or null when parsing fails.
 */
function clamp1to10(n) {
  const x = parseInt(n, 10);
  if (Number.isNaN(x)) return null;
  return Math.max(1, Math.min(10, x));
}

/**
 * Formats a score safely. Handles numbers and strings.
 * - Integers → "85"
 * - Floats  → "85.35" (2 decimal places)
 * - Invalid → "" (empty string) or fallback value
 * @param {(number|string|null|undefined)} score Raw score value.
 * @returns {string} Formatted score string, or "" when invalid/empty.
 */
function formatScore(score) {
  // Handle null, undefined, or empty
  if (score == null || score === '') {
    return '';
  }

  // Convert string to number if needed
  const num = typeof score === 'string' ? parseFloat(score) : Number(score);

  // If it's not a valid number
  if (isNaN(num)) {
    return '';
  }

  // Check if it's an integer
  if (Number.isInteger(num)) {
    return num.toString();
  }

  // It's a float → format to 2 decimal places
  return num.toFixed(2);
}

/**
 * Parses a JSON string, returning a fallback value on any parse error.
 * @param {string} str The JSON text to parse.
 * @param {*} fallback Value returned when parsing throws.
 * @returns {*} The parsed value, or fallback.
 */
function safeJsonParse(str, fallback) {
  try {
    return JSON.parse(str);
  } catch (_) {
    return fallback;
  }
}

/**
 * Builds a reasonably unique id by combining a prefix, the current
 * timestamp, and a random hex suffix.
 * @param {string} prefix Leading label for the id.
 * @returns {string} A unique-ish identifier string.
 */
function nowId(prefix) {
  return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

/**
 * Formats a duration in seconds as a compact "Nm Ns" (or "Ns") string.
 * @param {number} totalSeconds Duration in seconds; negatives treated as 0.
 * @returns {string} Human-readable duration label.
 */
function formatDuration(totalSeconds) {
  const s = Math.max(0, totalSeconds | 0);
  const m = Math.floor(s / 60);
  const r = s % 60;
  if (m <= 0) return `${r}s`;
  return `${m}m ${r}s`;
}

const DEFAULT_NOTE =
  'Seeds of Truth AI can make mistakes. Please verify important information.';

/**
 * Updates the input-area note text and toggles its busy/spinner styling.
 * @param {string} msg Message to display; falls back to DEFAULT_NOTE when empty.
 * @param {{busy?: boolean}=} options Options object.
 * @param {boolean=} options.busy Whether to show the busy state.
 * @returns {void}
 */
function setNoteMessage(msg, { busy = false } = {}) {
  if (els.noteMessage) els.noteMessage.textContent = msg || DEFAULT_NOTE;

  if (els.noteTextWrap) {
    els.noteTextWrap.classList.toggle('is-busy', !!busy);
  }

  // spinner visibility is controlled via .is-busy class
}

/**
 * Enables or disables the chat input and submit button to prevent
 * repeated submits while a request is in flight.
 * @param {boolean} isBusy Whether the UI should be in a busy state.
 * @returns {void}
 */
function setUiBusy(isBusy) {
  // prevent repeated submits + make it feel responsive
  if (els.chatInput) els.chatInput.disabled = !!isBusy;

  // if you have a submit button, disable it too (safe even if null)
  const submitBtn = els.chatForm?.querySelector('button[type="submit"]');
  if (submitBtn) submitBtn.disabled = !!isBusy;
}

/**
 * Starts a busy state and schedules staged "reassurance" status updates
 * for slow inference. Returns a cleanup function that MUST be called
 * (preferably in a finally block) to clear timers and end the busy state.
 * @param {string} opLabel Label for the operation in progress.
 * @returns {function(string=): void} A cleanup function; pass an optional
 *     final message to display when ending the busy state.
 */
function beginStatus(opLabel) {
  setUiBusy(true);
  setNoteMessage(opLabel, { busy: true });

  const timers = [];

  // staged “reassurance” updates for slow inference
  timers.push(
    setTimeout(() => setNoteMessage(`${opLabel}…`, { busy: true }), 800),
  );
  timers.push(
    setTimeout(() => setNoteMessage('Still working…', { busy: true }), 5000),
  );
  timers.push(
    setTimeout(
      () =>
        setNoteMessage('This can take ~30 seconds on some queries…', {
          busy: true,
        }),
      12000,
    ),
  );

  return function endStatus(finalMsg = null) {
    timers.forEach(clearTimeout);
    setUiBusy(false);
    setNoteMessage(finalMsg || DEFAULT_NOTE, { busy: false });
  };
}

/**
 * Returns a promise that resolves after the given delay.
 * @param {number} ms Delay in milliseconds.
 * @returns {Promise<void>} A promise resolved after the delay.
 */
function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

/**
 * Types fullText into textEl with a blinking cursor, character by
 * character. Uses textContent only (safe, no HTML injection); the first
 * maxTyped characters are animated and any remainder is snapped in.
 * @param {!Element} textEl Element to type the text into.
 * @param {string} fullText The full text to render.
 * @param {{cps?: number, chunkMin?: number, chunkMax?: number,
 *     maxTyped?: number}=} opts Typing options: characters-per-second
 *     target, min/max characters per chunk, and the cap on animated chars.
 * @returns {Promise<void>} Resolves when typing completes.
 */
async function typeIntoElement(textEl, fullText, opts = {}) {
  const {
    cps = 60, // characters per second target
    chunkMin = 1,
    chunkMax = 4,
    maxTyped = 800, // type first N chars then snap remainder (UX)
  } = opts;

  const text = String(fullText ?? '');
  const toType = text.slice(0, Math.min(text.length, maxTyped));
  const remainder = text.slice(toType.length);

  // Build cursor node
  const cursor = document.createElement('span');
  cursor.className = 'typing-cursor';
  cursor.textContent = '▍';

  // Clear and attach
  textEl.textContent = '';
  textEl.appendChild(cursor);

  let i = 0;

  while (i < toType.length) {
    const n = Math.min(
      toType.length - i,
      chunkMin + Math.floor(Math.random() * (chunkMax - chunkMin + 1)),
    );

    const slice = toType.slice(i, i + n);
    i += n;

    cursor.insertAdjacentText('beforebegin', slice);
    scrollChatToBottom?.();

    // pacing
    let ms = (1000 / cps) * n;
    const last = slice.slice(-1);
    if (last === '\n') ms += 120;
    else if ('.!?'.includes(last)) ms += 140;
    else if (',;:'.includes(last)) ms += 60;

    await sleep(ms);
  }

  // Snap remainder instantly (optional but recommended)
  if (remainder) cursor.insertAdjacentText('beforebegin', remainder);

  // Remove cursor when done
  cursor.remove();
  scrollChatToBottom?.();
}

/*------ SUBSET COMBO HELPER------------------*/
/**
 * Looks up the list of corpus subsets associated with a named combo.
 * @param {string} comboName Name of the subset combo (e.g. "Everything").
 * @returns {!Array<string>} The combo's subset names, or [] when not found.
 */
function getSubsetsForCombo(comboName) {
  const combo = SUBSET_COMBOS.find((c) => c.combo_name === comboName);
  return combo ? combo.subsets : [];
}
/*------ MARKDOWN HELPER------------------*/

/**
 * Escapes the five HTML-significant characters in a string.
 * @param {*} s Value to escape (coerced to string).
 * @returns {string} The HTML-escaped string.
 */
function escapeHtml(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/**
 * Renders a subset of Markdown to HTML. Safe-ish: the input is HTML-escaped
 * first, then only a limited set of tags is re-introduced.
 *
 * Inline elements (always): links, inline code, bold, italic, line breaks.
 * Block elements (opt-in via ``opts.blocks``): ATX headers (#..######),
 * unordered/ordered lists, blockquotes, fenced code blocks, and paragraphs.
 * Block mode is used for the chat/streaming reply; the lightweight inline-only
 * mode (default) preserves the original behavior for reference snippets.
 *
 * @param {*} md Plain-text Markdown input (coerced to string).
 * @param {{blocks?: boolean}} [opts] Set ``blocks: true`` to enable
 *   block-level rendering (headers, lists, blockquotes, fenced code).
 * @returns {string} The resulting HTML string.
 */
function renderMiniMarkdown(md, opts) {
  const useBlocks = !!(opts && opts.blocks);
  // Input should be plain text (NOT HTML). We'll return safe-ish HTML.
  let s = String(md ?? '');

  // Normalize newlines
  s = s.replace(/\r\n/g, '\n');

  // ---------- Escape HTML first ----------
  // (We will re-introduce only a/strong/em/code/br + block tags)
  s = s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

  // ---------- Helpers ----------
  // Use placeholders so later formatting doesn't touch generated <a> tags.
  const placeholders = [];
  const put = (html) => {
    const key = `\uE000${placeholders.length}\uE001`;
    placeholders.push(html);
    return key;
  };
  const restore = (text) =>
    text.replace(/\uE000(\d+)\uE001/g, (_, i) => placeholders[Number(i)] ?? '');

  const escapeAttr = (x) => String(x).replace(/"/g, '%22');

  // ---------- Fenced code blocks: ```lang\n...\n``` (blocks mode) ----------
  // Extracted first (as protected placeholders) so the inline rules below and
  // the block assembler don't touch code contents. The content is already
  // HTML-escaped above, so it's safe to drop straight into <pre><code>.
  if (useBlocks) {
    s = s.replace(/```[^\n]*\n([\s\S]*?)```/g, (m, code) =>
      put(`<pre class="md-pre"><code>${code.replace(/\n+$/, '')}</code></pre>`)
    );
  }

  // ---------- Markdown links: [text](url) and [text](<url>) ----------
  // Notes:
  // - We stop url at whitespace OR ')' but allow common URL chars.
  // - We also trim a trailing punctuation char if it "obviously" isn't part of the URL.
  const linkRe = /\[([^\]\n]+)\]\(\s*(<)?(https?:\/\/[^\s)]+)(>)?\s*\)/g;
  s = s.replace(linkRe, (m, text, _lt, url, _gt) => {
    let u = url;

    // Strip common trailing punctuation that is typically outside the URL
    // e.g. "...(https://x.com)." -> remove trailing '.'
    u = u.replace(/[.,;:!?]+$/g, '');

    const safeUrl = escapeAttr(u);
    const safeText = text; // already HTML-escaped above
    return put(
      `<a href="${safeUrl}" target="_blank" rel="noopener noreferrer">${safeText}</a>`,
    );
  });

  // ---------- Inline code: `code` ----------
  // Do this before bold/italic so markers inside code aren't parsed.
  s = s.replace(/`([^`\n]+)`/g, (m, code) => put(`<code>${code}</code>`));

  // ---------- Bold: **text** ----------
  // Use a non-greedy pattern and avoid crossing newlines.
  s = s.replace(/\*\*([^\n*][\s\S]*?[^\n*])\*\*/g, '<strong>$1</strong>');

  // ---------- Italic: *text* ----------
  // Avoid italicizing inside words and avoid matching **bold** fragments.
  // Also avoids matching lone '*' in things like pointers.
  s = s.replace(
    /(^|[\s(])\*([^\n*][\s\S]*?[^\n*])\*(?=[\s).,!?:;]|$)/g,
    '$1<em>$2</em>',
  );

  if (!useBlocks) {
    // ---------- Inline-only mode: newlines become <br> (original behavior) ----------
    s = s.replace(/\n/g, '<br>');
    return restore(s);
  }

  // ---------- Block assembly (headers, lists, blockquotes, paragraphs) ----------
  // Operates line-by-line on the inline-formatted text. Consecutive list /
  // blockquote lines are grouped; runs of plain lines become a <p> with <br>
  // joins; lone placeholders (e.g. fenced code blocks) pass through as blocks.
  const isPlaceholderOnly = (ln) => /^\d+$/.test(ln.trim());
  const reH = /^(#{1,6})\s+(.*)$/;
  const reQuote = /^\s*&gt;\s?/;      // '>' was escaped to '&gt;'
  const reUL = /^\s*[-*+]\s+(.*)$/;
  const reOL = /^\s*\d+\.\s+(.*)$/;

  const lines = s.split('\n');
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (trimmed === '') { i++; continue; }

    // Lone placeholder (fenced code block, etc.) -> emit as its own block.
    if (isPlaceholderOnly(line)) { out.push(trimmed); i++; continue; }

    // ATX header -> <h3>..<h6> (markdown level + 2, clamped) so a single '#'
    // doesn't render as an oversized page title inside a chat bubble.
    const h = reH.exec(line);
    if (h) {
      const lvl = Math.min(h[1].length + 2, 6);
      out.push(`<h${lvl} class="md-h">${h[2].trim()}</h${lvl}>`);
      i++; continue;
    }

    // Blockquote — gather consecutive '>' lines.
    if (reQuote.test(line)) {
      const quote = [];
      while (i < lines.length && reQuote.test(lines[i])) {
        quote.push(lines[i].replace(reQuote, ''));
        i++;
      }
      out.push('<blockquote class="md-quote">' + quote.join('<br>') + '</blockquote>');
      continue;
    }

    // Unordered list — gather consecutive '-', '*', '+' items.
    if (reUL.test(line)) {
      const items = [];
      while (i < lines.length && reUL.test(lines[i])) {
        items.push('<li>' + lines[i].replace(reUL, '$1').trim() + '</li>');
        i++;
      }
      out.push('<ul class="md-ul">' + items.join('') + '</ul>');
      continue;
    }

    // Ordered list — gather consecutive 'N.' items.
    if (reOL.test(line)) {
      const items = [];
      while (i < lines.length && reOL.test(lines[i])) {
        items.push('<li>' + lines[i].replace(reOL, '$1').trim() + '</li>');
        i++;
      }
      out.push('<ol class="md-ol">' + items.join('') + '</ol>');
      continue;
    }

    // Paragraph — gather consecutive plain lines until a blank line or a line
    // that starts a different block.
    const para = [];
    while (
      i < lines.length &&
      lines[i].trim() !== '' &&
      !isPlaceholderOnly(lines[i]) &&
      !reH.test(lines[i]) &&
      !reQuote.test(lines[i]) &&
      !reUL.test(lines[i]) &&
      !reOL.test(lines[i])
    ) {
      para.push(lines[i]);
      i++;
    }
    if (para.length) out.push('<p class="md-p">' + para.join('<br>') + '</p>');
  }

  return restore(out.join('\n'));
}

/**
 * Decodes HTML entities (e.g. &amp;, &#39;, numeric refs) to plain text.
 * @param {*} str String containing HTML entities (coerced to string).
 * @returns {string} The decoded text.
 */
function decodeHtmlEntities(str) {
  // Decodes &#...; &amp; &quot; etc
  const txt = document.createElement('textarea');
  txt.innerHTML = String(str ?? '');
  return txt.value;
}

/**
 * Strips common LLM stop tokens / scaffolding (e.g. <|im_end|>, </s>) and
 * normalizes whitespace from a RAG-generated string.
 * @param {*} str Raw model output (coerced to string).
 * @returns {string} The cleaned, trimmed string.
 */
function cleanRagArtifacts(str) {
  let s = String(str ?? '');

  // Common LLM stop tokens / scaffolding
  s = s.replace(/<\|im_end\|>/g, '');
  s = s.replace(/<\/s>/g, '');
  s = s.replace(/<\|endoftext\|>/g, '');

  // Sometimes these appear HTML-escaped already
  s = s.replace(/&lt;\|im_end\|&gt;/g, '');
  s = s.replace(/&lt;\/s&gt;/g, '');
  s = s.replace(/&lt;\|endoftext\|&gt;/g, '');

  // Collapse weird whitespace
  s = s.replace(/\r\n/g, '\n');
  s = s.replace(/[ \t]+\n/g, '\n');
  s = s.replace(/\n{3,}/g, '\n\n');

  return s.trim();
}

/* =========================================================
   4) MODAL (custom alert/confirm)
   ========================================================= */
let modalResolve = null;

/**
 * Populates and shows the custom modal overlay with the given content and
 * buttons, focuses the first button, and wires the Escape-key handler.
 * @param {{title: string, message: string,
 *     buttons: !Array<{label: string, value: *, variant?: string}>}} options
 *     Modal content: title, message, and the action buttons to render.
 * @returns {void}
 */
function openModal({ title, message, buttons }) {
  els.modalTitle.textContent = title || 'Notice';
  els.modalMessage.textContent = message || '';
  els.modalActions.innerHTML = '';

  buttons.forEach((b) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `modal-btn${b.variant ? ' ' + b.variant : ''}`;
    btn.textContent = b.label;
    btn.addEventListener('click', () => closeModal(b.value));
    els.modalActions.appendChild(btn);
  });

  els.modalOverlay.classList.add('show');
  els.modalOverlay.setAttribute('aria-hidden', 'false');

  const firstBtn = els.modalActions.querySelector('button');
  if (firstBtn) firstBtn.focus();

  document.addEventListener('keydown', onModalKeydown);
}

/**
 * Keydown handler for the modal: closes it (with false) on Escape.
 * @param {!KeyboardEvent} e The keyboard event.
 * @returns {void}
 */
function onModalKeydown(e) {
  if (e.key === 'Escape') closeModal(false);
}

/**
 * Hides the custom modal overlay, clears its actions, and resolves the
 * pending modal promise (if any) with the given result.
 * @param {*} result Value to resolve the modal promise with.
 * @returns {void}
 */
function closeModal(result) {
  if (!els.modalOverlay) return;
  els.modalOverlay.classList.remove('show');
  els.modalOverlay.setAttribute('aria-hidden', 'true');
  els.modalActions.innerHTML = '';

  const resolve = modalResolve;
  modalResolve = null;
  if (resolve) resolve(result);

  document.removeEventListener('keydown', onModalKeydown);
}

/**
 * Opens a confirm-style modal and resolves with the user's choice.
 * @param {{title?: string, message?: string, confirmText?: string,
 *     cancelText?: string, danger?: boolean}=} options Modal text and the
 *     danger flag for styling the confirm button.
 * @returns {Promise<boolean>} Resolves true if confirmed, false otherwise.
 */
function modalConfirm({
  title = 'Confirm',
  message = 'Are you sure?',
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  danger = false,
} = {}) {
  return new Promise((resolve) => {
    modalResolve = resolve;
    openModal({
      title,
      message,
      buttons: [
        { label: cancelText, value: false },
        {
          label: confirmText,
          value: true,
          variant: danger ? 'danger' : 'primary',
        },
      ],
    });
  });
}

/**
 * Opens an alert-style modal with a single OK button.
 * @param {{title?: string, message?: string, okText?: string}=} options
 *     Modal text content.
 * @returns {Promise<boolean>} Resolves true when the modal is dismissed.
 */
function modalAlert({ title = 'Notice', message = '', okText = 'OK' } = {}) {
  return new Promise((resolve) => {
    modalResolve = resolve;
    openModal({
      title,
      message,
      buttons: [{ label: okText, value: true, variant: 'primary' }],
    });
  });
}

/* =========================================================
   5) THEME
   ========================================================= */
/**
 * Applies the given theme to the document body and persists it.
 * @param {string} mode Theme mode, 'light' or 'dark'.
 * @returns {void}
 */
function applyTheme(mode) {
  if (mode === 'light') els.body.classList.add('light');
  else els.body.classList.remove('light');
  try {
    localStorage.setItem(CFG.LS_THEME, mode);
  } catch (_) {}
}

/**
 * Initializes the theme from localStorage (falling back to the default)
 * and wires the theme-toggle buttons.
 * @returns {void}
 */
function initTheme() {
  let mode = CFG.DEFAULT_THEME;
  try {
    const stored = localStorage.getItem(CFG.LS_THEME);
    if (stored === 'light' || stored === 'dark') mode = stored;
  } catch (_) {}
  applyTheme(mode);

  els.themeToggleButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const isLightNow = els.body.classList.contains('light');
      applyTheme(isLightNow ? 'dark' : 'light');
    });
  });
}

/* =========================================================
   6) SIDEBAR COLLAPSE / MOBILE MENU
   ========================================================= */
/**
 * Collapses or expands the sidebar and persists the state.
 * @param {boolean} collapsed Whether the sidebar should be collapsed.
 * @returns {void}
 */
function setSidebarCollapsed(collapsed) {
  els.body.classList.toggle('sidebar-collapsed', !!collapsed);
  try {
    localStorage.setItem(CFG.LS_SIDEBAR_COLLAPSED, collapsed ? '1' : '0');
  } catch (_) {}
}

/**
 * Restores the persisted sidebar-collapsed state and wires the
 * collapse/open buttons.
 * @returns {void}
 */
function initSidebarCollapse() {
  try {
    if (localStorage.getItem(CFG.LS_SIDEBAR_COLLAPSED) === '1')
      setSidebarCollapsed(true);
  } catch (_) {}

  if (els.sidebarCollapseBtn)
    els.sidebarCollapseBtn.addEventListener('click', () =>
      setSidebarCollapsed(true),
    );
  if (els.sidebarOpenBtn)
    els.sidebarOpenBtn.addEventListener('click', () =>
      setSidebarCollapsed(false),
    );
}

/**
 * Wires the mobile menu button and overlay to toggle the off-canvas
 * sidebar on small screens.
 * @returns {void}
 */
function initMobileSidebar() {
  if (els.menuBtn) {
    els.menuBtn.addEventListener('click', () => {
      els.sidebar.classList.toggle('visible');
      els.overlay.classList.toggle('visible');
    });
  }
  if (els.overlay) {
    els.overlay.addEventListener('click', () => {
      els.sidebar.classList.remove('visible');
      els.overlay.classList.remove('visible');
    });
  }
}

/* =========================================================
   7) TOOLS POPUP + TOOL STATE
   ========================================================= */

/**
 * Builds the tool-related portion of an API request payload from the
 * current toolState, expanding the selected topic combo into subsets and
 * pinning algo/prompt/force-queue values appropriately for DEV_MODE.
 * @returns {{use_rag: boolean, subsets: !Array<string>,
 *     rag_algo_type: number, prompt_type: number, force_queue?: string}}
 *     The tool payload fragment to merge into the request body.
 */
function getToolPayloadState() {
  const selectedSubsets = getSubsetsForCombo(toolState.searchGroup);

  return {
    use_rag: !!toolState.useRag,

    // Always send selected subsets.
    // Backend can ignore these when use_rag is false.
    subsets: selectedSubsets,

    // In DEV_MODE the user picks these via the (visible) Algo/Prompt inputs.
    // Otherwise force the production defaults — the inputs are hidden and any
    // value left over in toolState/localStorage from dev sessions is ignored.
    rag_algo_type: CFG.DEV_MODE
      ? clampInt(toolState.ragAlgoType, 1, 10, CFG.DEFAULT_RAG_ALGO_TYPE)
      : CFG.DEFAULT_RAG_ALGO_TYPE,
    prompt_type: CFG.DEV_MODE
      ? clampInt(toolState.promptType, 1, 10, CFG.DEFAULT_PROMPT_TYPE)
      : CFG.DEFAULT_PROMPT_TYPE,

    // Dev-only knob. Server reads it as a string-coerced bool
    // (utils.str_to_bool). Only sent in DEV_MODE so any leftover LS state
    // from a dev session can't escape into a prod tab.
    ...(CFG.DEV_MODE && toolState.forceQueue ? { force_queue: 'true' } : {}),
  };
}

/**
 * Loads persisted tool settings from localStorage into toolState,
 * validating and clamping each field, and pinning hidden controls to
 * production defaults when not in DEV_MODE.
 * @returns {void}
 */
function loadToolState() {
  try {
    // Fall back to an empty object when nothing is saved yet, so the
    // non-DEV_MODE default-pinning below still runs on a fresh browser.
    const raw = localStorage.getItem(CFG.LS_TOOLS) || '{}';

    const parsed = safeJsonParse(raw, {});

    if (typeof parsed.historyTurns === 'number') {
      toolState.historyTurns = clampInt(
        parsed.historyTurns,
        0,
        CFG.MAX_CONTEXT_TURNS,
        CFG.DEFAULT_HISTORY_TURNS,
      );
    }

    if (['search', 'chat', 'ab'].includes(parsed.mode))
      toolState.mode = parsed.mode;

    // NEW: RAG toggle (default ON if missing)
    if (typeof parsed.useRag === 'boolean') {
      toolState.useRag = parsed.useRag;
    } else if (typeof toolState.useRag !== 'boolean') {
      toolState.useRag = true;
    }
    if (typeof parsed.searchGroup === 'string') {
      toolState.searchGroup = parsed.searchGroup;
    }
    const savedRagAlgoType = clamp1to10(parsed.ragAlgoType);
    if (savedRagAlgoType !== null) {
      toolState.ragAlgoType = savedRagAlgoType;
    }

    const savedPromptType = clamp1to10(parsed.promptType);
    if (savedPromptType !== null) {
      toolState.promptType = savedPromptType;
    }

    // Dev-only force-queue toggle. Always rehydrate as false outside
    // DEV_MODE so a previous dev-mode session can't leak this on.
    if (CFG.DEV_MODE && typeof parsed.forceQueue === 'boolean') {
      toolState.forceQueue = parsed.forceQueue;
    } else {
      toolState.forceQueue = false;
    }

    // Outside DEV_MODE the Model / Memory / Retrieval controls are
    // hidden (via data-dev-only) and the user cannot change them. Pin them
    // to fixed prod defaults so any leftover dev-session localStorage can't
    // leak non-default values into requests. Mode and Select Topic stay
    // user-controlled — both are visible in every mode.
    if (!CFG.DEV_MODE) {
      toolState.historyTurns = 0;
      toolState.useRag = true;
      toolState.searchGroup = 'Deep Politics';
      toolState.modelType = 'spark';
    }
  } catch (_) {}
}

/**
 * Persists the current toolState to localStorage.
 * @returns {void}
 */
function saveToolState() {
  // NEW: keep backwards compatibility + ensure useRag exists
  if (typeof toolState.useRag !== 'boolean') toolState.useRag = true;
  try {
    localStorage.setItem(CFG.LS_TOOLS, JSON.stringify(toolState));
  } catch (_) {}
}

/**
 * Syncs the tools-popup DOM controls (history slider, mode radios, help
 * text, RAG/force-queue toggles, topic select, algo/prompt inputs) to the
 * current toolState. Also downgrades mode to 'search' when locked.
 * @returns {void}
 */
function renderToolState() {
  if (els.historySlider)
    els.historySlider.value = String(toolState.historyTurns);
  if (els.historyValue)
    els.historyValue.textContent = String(toolState.historyTurns);
  if (els.historyHelpN)
    els.historyHelpN.textContent = String(toolState.historyTurns);

  // if locked, force search
  if (!isUnlocked && (toolState.mode === 'chat' || toolState.mode === 'ab')) {
    toolState.mode = 'search';
  }

  const id =
    toolState.mode === 'search'
      ? 'mode-search'
      : toolState.mode === 'ab'
        ? 'mode-ab'
        : 'mode-chat';
  const el = document.getElementById(id);
  if (el) el.checked = true;

  if (els.modeHelp) {
    els.modeHelp.textContent =
      toolState.mode === 'search'
        ? 'Search the corpus without AI'
        : toolState.mode === 'ab'
          ? 'A/B test two responses and select the best one'
          : 'AI chat: normal chat mode';
  }

  if (els.referencesTitle) {
    els.referencesTitle.textContent =
      toolState.mode === 'search' ? 'Search Results' : 'References';
  }

  // NEW: render RAG toggle state (do not affect conversation memory)
  const ragToggle = els.ragToggle || document.getElementById('rag-toggle');
  if (ragToggle) ragToggle.checked = !!toolState.useRag;

  // Dev-only force-queue toggle. Outside DEV_MODE the checkbox is hidden
  // (via data-dev-only) and toolState.forceQueue is forced false on load,
  // so nothing to do here in that case.
  const forceQueueToggle =
    els.forceQueueToggle || document.getElementById('force-queue-toggle');
  if (forceQueueToggle) forceQueueToggle.checked = !!toolState.forceQueue;

  // Optional: update help text dynamically if you included #rag-help
  const ragHelp = document.getElementById('rag-help');
  if (ragHelp) {
    ragHelp.textContent = toolState.useRag
      ? 'Uses retrieved context (RAG) to ground responses.'
      : 'Model-only: skips retrieval, but still uses conversation memory.';
  }
  if (els.searchGroup) {
    els.searchGroup.value = toolState.searchGroup;
  }
  if (els.ragAlgoType) {
    els.ragAlgoType.value = String(toolState.ragAlgoType);
  }

  if (els.promptType) {
    els.promptType.value = String(toolState.promptType);
  }
}

/**
 * Initializes the tools popup: open/close behavior and change handlers
 * for the history slider, mode radios, RAG toggle, dev-only force-queue
 * toggle, topic select, and the algo/prompt number inputs.
 * @returns {void}
 */
function initToolsPopup() {
  if (!els.toolsBtn || !els.toolsPopup) return;

  // Ensure we have the rag toggle element (works even if you didn't add to els)
  if (!els.ragToggle) els.ragToggle = document.getElementById('rag-toggle');

  els.toolsBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    els.toolsPopup.classList.toggle('visible');
  });

  document.addEventListener('click', (e) => {
    if (!els.toolsPopup.contains(e.target) && e.target !== els.toolsBtn) {
      els.toolsPopup.classList.remove('visible');
    }
  });

  // slider change
  if (els.historySlider) {
    els.historySlider.addEventListener('input', () => {
      toolState.historyTurns = clampInt(
        els.historySlider.value,
        0,
        CFG.MAX_CONTEXT_TURNS,
        CFG.DEFAULT_HISTORY_TURNS,
      );
      renderToolState();
      saveToolState();
    });
  }

  // radios change
  if (els.modeRadios && els.modeRadios.length) {
    els.modeRadios.forEach((r) => {
      r.addEventListener('change', () => {
        if (!r.checked) return;

        const val = r.value;
        if (!isUnlocked && (val === 'chat' || val === 'ab')) {
          // bounce back to search
          const search = document.getElementById('mode-search');
          if (search) search.checked = true;
          toolState.mode = 'search';
          renderToolState();
          saveToolState();
          pushStatusMessage('Locked: search mode only.');
          return;
        }

        toolState.mode = val;
        renderToolState();
        saveToolState();
      });
    });
  }

  // NEW: RAG toggle change
  if (els.ragToggle) {
    els.ragToggle.addEventListener('change', () => {
      toolState.useRag = !!els.ragToggle.checked;
      renderToolState(); // keeps help text in sync, harmless otherwise
      saveToolState();

      // Optional feedback
      if (typeof pushStatusMessage === 'function') {
        pushStatusMessage(
          toolState.useRag ? 'RAG enabled.' : 'RAG disabled (model-only).',
        );
      }
    });
  }

  // Dev-only: Force-queue toggle. The checkbox itself is hidden via the
  // data-dev-only attribute when DEV_MODE is off, but we still gate the
  // wiring here as a second layer so a leftover LS value can't sneak the
  // flag into prod payloads.
  if (els.forceQueueToggle && CFG.DEV_MODE) {
    els.forceQueueToggle.addEventListener('change', () => {
      toolState.forceQueue = !!els.forceQueueToggle.checked;
      saveToolState();

      if (typeof pushStatusMessage === 'function') {
        pushStatusMessage(
          toolState.forceQueue
            ? 'Force-queue ON: next chat will be queued.'
            : 'Force-queue OFF.',
        );
      }
    });
  }
  // Model selection change
  if (modelSelect && toolState.modelType) {
    modelSelect.value = toolState.modelType;
  }
  // Search groups change
  if (els.searchGroup) {
    els.searchGroup.addEventListener('change', () => {
      toolState.searchGroup = els.searchGroup.value;
      saveToolState();

      if (typeof pushStatusMessage === 'function') {
        pushStatusMessage(`Search group: ${toolState.searchGroup}`);
      }
    });
  }
  /**
   * Wires a number input so that change/blur clamp its value into [1, 10],
   * write it into toolState, persist, and emit a status message.
   * @param {?Element} el The number input element (may be null).
   * @param {string} stateKey Key in toolState to update.
   * @param {string} label Human-readable label for the status message.
   * @returns {void}
   */
  function bindToolNumberInput(el, stateKey, label) {
    if (!el) return;

    const sync = () => {
      const value = clampInt(el.value, 1, 10, 1);
      toolState[stateKey] = value;
      el.value = String(value);
      saveToolState();

      if (typeof pushStatusMessage === 'function') {
        pushStatusMessage(`${label}: ${value}`);
      }
    };

    el.addEventListener('change', sync);
    el.addEventListener('blur', sync);
  }

  bindToolNumberInput(els.ragAlgoType, 'ragAlgoType', 'Algo');
  bindToolNumberInput(els.promptType, 'promptType', 'Prompt');
}

// Model selection
const modelSelect = document.getElementById('model-type');

if (modelSelect) {
  modelSelect.addEventListener('change', () => {
    toolState.modelType = modelSelect.value;
    saveToolState();
  });
}

/* =========================================================
   8) STATUS PANEL
   ========================================================= */
/**
 * Updates the endpoint status indicator (dot color, label, and chip text).
 * @param {string} status One of 'off', 'starting', 'ready', or any other
 *     value for an unknown/"checking" state.
 * @returns {void}
 */
function setEndpointStatus(status) {
  if (!els.endpointDot) return;

  els.endpointDot.classList.remove('red', 'yellow', 'green');

  if (status === 'off') {
    els.endpointDot.classList.add('red');
    els.endpointLabel.textContent = 'Endpoint offline';
    els.endpointChip.textContent = 'offline';
  } else if (status === 'starting') {
    els.endpointDot.classList.add('yellow');
    els.endpointLabel.textContent = 'Endpoint starting…';
    els.endpointChip.textContent = 'starting';
  } else if (status === 'ready') {
    els.endpointDot.classList.add('green');
    els.endpointLabel.textContent = 'Endpoint ready';
    els.endpointChip.textContent = 'ready';
  } else {
    els.endpointLabel.textContent = 'Checking endpoint…';
    els.endpointChip.textContent = 'unknown';
  }
}

/**
 * Updates the queue count display and its estimated wait time (assuming
 * roughly 45 seconds per queued query).
 * @param {(number|string)} queriesInLine Number of queries waiting.
 * @returns {void}
 */
function setQueueStatus(queriesInLine) {
  const q = Math.max(0, parseInt(queriesInLine, 10) || 0);
  if (els.queueCountEl) els.queueCountEl.textContent = String(q);
  if (els.queueEtaEl) els.queueEtaEl.textContent = formatDuration(q * 45);
}

/**
 * Prepends a timestamped message to the status panel, keeping only the
 * most recent message visible.
 * @param {string} text Message text; ignored when empty.
 * @returns {void}
 */
function pushStatusMessage(text) {
  const msg = String(text || '').trim();
  if (!msg || !els.statusMessagesEl) return;

  const placeholder = els.statusMessagesEl.querySelector(
    '.status-message.muted',
  );
  if (placeholder) placeholder.remove();

  const el = document.createElement('div');
  el.className = 'status-message';
  const ts = new Date().toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });
  el.textContent = `[${ts}] ${msg}`;
  els.statusMessagesEl.prepend(el);

  const items = els.statusMessagesEl.querySelectorAll('.status-message');
  if (items.length > 1) items[items.length - 1].remove();
}

/* =========================================================
   9) FEEDBACK MODAL (local save for now)
   ========================================================= */
/**
 * Briefly shows the feedback confirmation toast, then hides it.
 * @returns {void}
 */
function showToast() {
  if (!els.fbToast) return;
  els.fbToast.classList.add('show');
  els.fbToast.setAttribute('aria-hidden', 'false');
  setTimeout(() => {
    els.fbToast.classList.remove('show');
    els.fbToast.setAttribute('aria-hidden', 'true');
  }, 1400);
}

/**
 * Opens the feedback modal for a given target (a chat response or a
 * reference), resetting the rating inputs and showing/hiding the
 * accuracy/style fields depending on the target type.
 * @param {{type: string, id?: string, snippet?: string, job_id?: string}}
 *     target The feedback target descriptor.
 * @returns {void}
 */
function openFeedbackModal(target) {
  const isRef = target?.type === 'reference';
  feedbackTarget = target;

  // reset
  if (els.fbAccuracy) els.fbAccuracy.value = 8;
  if (els.fbStyle) els.fbStyle.value = 8;
  if (els.fbRelevance) els.fbRelevance.value = 8;
  if (els.fbComments) els.fbComments.value = '';

  if (els.fbFieldAccuracy)
    els.fbFieldAccuracy.style.display = isRef ? 'none' : '';
  if (els.fbFieldStyle) els.fbFieldStyle.style.display = isRef ? 'none' : '';
  if (els.fbJobId) els.fbJobId.value = target?.job_id;

  const label = isRef ? 'Reference' : 'Response';
  const snip = (target?.snippet || '').trim().replace(/\s+/g, ' ');
  const short = snip.length > 120 ? snip.slice(0, 120) + '…' : snip;
  if (els.fbMeta)
    els.fbMeta.textContent = `${label} ID: ${target?.id || 'n/a'}${short ? ' — ' + short : ''}`;

  els.fbOverlay.classList.add('show');
  els.fbOverlay.setAttribute('aria-hidden', 'false');

  (isRef ? els.fbRelevance : els.fbAccuracy)?.focus?.();
}

/**
 * Hides the feedback modal and clears the current feedback target.
 * @returns {void}
 */
function closeFeedbackModal() {
  if (!els.fbOverlay) return;
  els.fbOverlay.classList.remove('show');
  els.fbOverlay.setAttribute('aria-hidden', 'true');
  feedbackTarget = null;
}

/**
 * Stores a feedback payload at the front of the localStorage feedback
 * list, trimming the list to MAX_FEEDBACK_ITEMS.
 * @param {!Object} payload The feedback record to store.
 * @returns {void}
 */
function saveFeedbackLocally(payload) {
  try {
    const arr = safeJsonParse(
      localStorage.getItem(CFG.LS_FEEDBACK) || '[]',
      [],
    );
    arr.unshift(payload);
    if (arr.length > CFG.MAX_FEEDBACK_ITEMS)
      arr.length = CFG.MAX_FEEDBACK_ITEMS;
    localStorage.setItem(CFG.LS_FEEDBACK, JSON.stringify(arr));
  } catch (_) {}
}

/**
 * Reads and validates the feedback form, saves the feedback locally, and
 * (when unlocked, in chat/ab mode, with a valid job id) sends it to the
 * server. Shows validation alerts on missing ratings.
 * @returns {Promise<void>} Resolves when the submission flow completes.
 */
async function submitFeedback() {
  if (!feedbackTarget) return;

  const isRef = feedbackTarget?.type === 'reference';

  const relevance = clamp1to10(els.fbRelevance.value);
  const accuracy = isRef ? undefined : clamp1to10(els.fbAccuracy.value);
  const style = isRef ? undefined : clamp1to10(els.fbStyle.value);
  const comments = (els.fbComments.value || '').trim();
  const job_id = (els.fbJobId?.value ?? '').trim();

  const payload = {
    target: {
      type: feedbackTarget?.type || 'unknown',
      id: feedbackTarget?.id || null,
      snippet: feedbackTarget?.snippet || '',
    },
    ratings: {
      relevance,
      ...(isRef ? {} : { accuracy, style }),
    },
    comments,
    createdAt: Date.now(),
  };

  if (!relevance) {
    await modalAlert({
      title: 'Missing rating',
      message: 'Please enter 1–10 for Relevance.',
    });
    return;
  }
  if (!isRef && (!accuracy || !style)) {
    await modalAlert({
      title: 'Missing ratings',
      message: 'Please enter 1–10 for Accuracy and Style.',
    });
    return;
  }

  // save feedback locally
  saveFeedbackLocally(payload);

  // send feedback only if unlocked and using chat mode
  // fyi we need a valid job id in order to send a feedback request to the server
  if (
    isUnlocked &&
    (toolState.mode === 'chat' || toolState.mode === 'ab') &&
    job_id != CFG.JOB_ID_NONE
  ) {
    await sendFeedback(job_id, relevance, accuracy, style, comments);
  }

  closeFeedbackModal();
  showToast();
}

/**
 * POSTs a feedback record to the server's feedback endpoint.
 * @param {string} job_id The job id the feedback is about.
 * @param {number} relevance Relevance rating (1-10).
 * @param {(number|undefined)} accuracy Accuracy rating (1-10), if any.
 * @param {(number|undefined)} style Style rating (1-10), if any.
 * @param {string=} comments Free-text comments.
 * @returns {Promise<Object>} The parsed JSON response.
 */
async function sendFeedback(job_id, relevance, accuracy, style, comments = '') {
  const res = await fetch(CFG.API.FEEDBACK, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      job_id,
      relevance,
      accuracy,
      style,
      comments,
    }),
  });

  const data = await res.json();

  if (!res.ok) {
    const errorMsg =
      'send feedback back failed: ' + (data?.error || 'request failed');
    pushStatusMessage(errorMsg);
  }

  return data;
}

/**
 * Wires the feedback modal's close/cancel/submit buttons, overlay-click
 * dismissal, and Escape-key handling.
 * @returns {void}
 */
function initFeedbackModal() {
  if (els.fbClose) els.fbClose.addEventListener('click', closeFeedbackModal);
  if (els.fbCancel) els.fbCancel.addEventListener('click', closeFeedbackModal);
  if (els.fbSubmit) els.fbSubmit.addEventListener('click', submitFeedback);

  if (els.fbOverlay) {
    els.fbOverlay.addEventListener('click', (e) => {
      if (e.target === els.fbOverlay) closeFeedbackModal();
    });
    document.addEventListener('keydown', (e) => {
      if (els.fbOverlay.classList.contains('show') && e.key === 'Escape')
        closeFeedbackModal();
    });
  }
}

/*--------- Copy to clipboard helper  --------------*/

/**
 * Copies text to the clipboard, using the async Clipboard API in secure
 * contexts and falling back to a hidden textarea + execCommand otherwise.
 * @param {*} text Text to copy (coerced to string).
 * @returns {Promise<boolean>} Resolves true on success, false on failure
 *     or when the text is empty.
 */
async function copyTextToClipboard(text) {
  const s = String(text ?? '');
  if (!s) return false;

  // Preferred (secure context: https / localhost)
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(s);
    return true;
  }

  // Fallback
  const ta = document.createElement('textarea');
  ta.value = s;
  ta.setAttribute('readonly', '');
  ta.style.position = 'fixed';
  ta.style.left = '-9999px';
  ta.style.top = '0';
  document.body.appendChild(ta);
  ta.select();

  const ok = document.execCommand('copy');
  ta.remove();
  return ok;
}

/**
 * Temporarily swaps a button's tooltip text, restoring it after a delay.
 * @param {?Element} btn Button element with a data-tooltip attribute.
 * @param {string=} msg Temporary tooltip text.
 * @param {number=} ms How long to show the temporary text, in ms.
 * @returns {void}
 */
function bumpTooltip(btn, msg = 'Copied!', ms = 900) {
  if (!btn?.dataset) return;
  const prev = btn.dataset.tooltip;
  btn.dataset.tooltip = msg;
  setTimeout(() => (btn.dataset.tooltip = prev), ms);
}

/* =========================================================
   10) CHAT UI RENDERING
   ========================================================= */
/**
 * Scrolls the chat container to the bottom.
 * @returns {void}
 */
function scrollChatToBottom() {
  if (els.chatContainer)
    els.chatContainer.scrollTop = els.chatContainer.scrollHeight;
}

/**
 * Fills an avatar element with the bot logo image.
 * @param {!Element} el The avatar element to populate.
 * @returns {void}
 */
function setBotAvatar(el) {
  el.textContent = ''; // remove any text
  el.innerHTML = `
    <img src="/static/img/logo512.png" alt="" class="message-avatar-img" />
  `;
}

/**
 * Appends a chat message row to the messages list. Bot messages include a
 * feedback button; the job id is stored on the text element's dataset.
 * @param {string} text Message body text.
 * @param {string} role Either 'bot' or 'user'.
 * @param {string=} job_id Associated job id (defaults to JOB_ID_NONE).
 * @returns {void}
 */
function appendMessage(text, role, job_id = CFG.JOB_ID_NONE) {
  const row = document.createElement('div');
  row.className = 'message-row ' + (role === 'bot' ? 'bot' : 'user');

  const inner = document.createElement('div');
  inner.className = 'message-content';

  const avatar = document.createElement('div');
  avatar.className = 'message-avatar ' + (role === 'bot' ? 'bot' : 'user');
  if (role === 'bot') {
    setBotAvatar(avatar);
  } else {
    avatar.textContent = 'You';
  }

  const textEl = document.createElement('div');
  textEl.className = 'message-text';
  textEl.textContent = text;

  // attach job_id
  textEl.dataset.jobId = job_id;

  inner.appendChild(avatar);

  if (role === 'bot') {
    const msgId = `bot_${++botMsgCounter}`;

    const contentWrap = document.createElement('div');
    contentWrap.style.display = 'flex';
    contentWrap.style.alignItems = 'flex-start';
    contentWrap.style.gap = '10px';
    contentWrap.style.width = '100%';

    textEl.style.flex = '1';

    const actions = document.createElement('div');
    actions.className = 'msg-actions';

    const commentBtn = document.createElement('button');
    commentBtn.type = 'button';
    commentBtn.className = 'comment-btn has-tooltip';
    commentBtn.dataset.tooltip = 'Add feedback';
    commentBtn.setAttribute('aria-label', 'Add feedback');
    commentBtn.innerHTML = `
      <svg viewBox="0 0 24 24" width="18" height="18" fill="none"
        stroke="currentColor" stroke-width="2.2"
        stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z"/>
      </svg>
    `;

    commentBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      openFeedbackModal({ type: 'response', id: msgId, snippet: text, job_id });
    });

    actions.appendChild(commentBtn);
    contentWrap.appendChild(textEl);
    contentWrap.appendChild(actions);
    inner.appendChild(contentWrap);
  } else {
    inner.appendChild(textEl);
  }

  row.appendChild(inner);
  els.messagesEl.appendChild(row);
  scrollChatToBottom();
}

/**
 * Renders an A/B-mode bot message: two side-by-side answer panels, each
 * selectable and with copy + feedback buttons, then types both answers in.
 * @param {string} aText Text of response A.
 * @param {string} bText Text of response B.
 * @param {string} job_id_a Job id for response A.
 * @param {string} job_id_b Job id for response B.
 * @param {{labelA?: string, labelB?: string}=} meta Optional panel labels.
 * @returns {Promise<void>} Resolves when both answers finish typing.
 */
async function appendABMessage(aText, bText, job_id_a, job_id_b, meta = {}) {
  const row = document.createElement('div');
  row.className = 'message-row bot';

  const inner = document.createElement('div');
  inner.className = 'message-content';

  const avatar = document.createElement('div');
  avatar.className = 'message-avatar bot';
  setBotAvatar(avatar);

  const wrap = document.createElement('div');
  wrap.className = 'ab-wrap';

  /**
   * Marks one A/B panel as the selected response, clearing the others.
   * @param {!Element} panel The panel element to select.
   * @param {string} variant The variant label ('A' or 'B').
   * @returns {void}
   */
  function selectPanel(panel, variant) {
    wrap.querySelectorAll('.ab-panel').forEach((p) => {
      p.classList.remove('selected');
      const btn = p.querySelector('.ab-select-btn');
      if (btn) btn.innerHTML = 'Select this response';
    });

    panel.classList.add('selected');

    const btn = panel.querySelector('.ab-select-btn');
    if (btn) btn.innerHTML = '✓ Selected';
  }

  /**
   * Builds one A/B answer panel with header, copy/feedback buttons, an
   * empty body for typed text, and a footer select button.
   * @param {string} label Display label for the panel.
   * @param {string} variant Variant identifier ('A' or 'B').
   * @param {string} job_id Job id used when opening the feedback modal.
   * @returns {{panel: !Element, body: !Element,
   *     setSnippet: function(string): void}} The panel element, its body
   *     element, and a setter to record the final text for copy/feedback.
   */
  function makePanel(label, variant, job_id) {
    const panel = document.createElement('div');
    panel.className = 'ab-panel';

    const head = document.createElement('div');
    head.className = 'ab-head';

    const lbl = document.createElement('div');
    lbl.className = 'ab-label';
    lbl.textContent = label;

    const actions = document.createElement('div');
    actions.className = 'ab-actions';

    const id = `ab_${variant}_${nowId('msg')}`;

    // Track final text
    let snippetForFeedback = `[A/B ${variant}]`;
    let latestTextForCopy = '';

    /* COPY BUTTON */
    const copyBtn = document.createElement('button');
    copyBtn.type = 'button';
    copyBtn.className = 'copy-btn comment-btn has-tooltip';
    copyBtn.dataset.tooltip = 'Copy to clipboard';
    copyBtn.setAttribute('aria-label', 'Copy response to clipboard');

    copyBtn.innerHTML = `
      <svg viewBox="0 0 24 24" width="18" height="18" fill="none"
        stroke="currentColor" stroke-width="2.2"
        stroke-linecap="round" stroke-linejoin="round">
        <rect x="9" y="9" width="13" height="13" rx="2"></rect>
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
      </svg>
    `;

    copyBtn.addEventListener('click', async (e) => {
      e.stopPropagation();
      try {
        const textToCopy = latestTextForCopy || body.textContent || '';
        const ok = await copyTextToClipboard(textToCopy);
        bumpTooltip(copyBtn, ok ? 'Copied!' : 'Copy failed');
      } catch {
        bumpTooltip(copyBtn, 'Copy failed');
      }
    });

    /* FEEDBACK BUTTON */
    const commentBtn = document.createElement('button');
    commentBtn.type = 'button';
    commentBtn.className = 'comment-btn';
    commentBtn.setAttribute('aria-label', 'Add feedback');

    commentBtn.innerHTML = `
      <svg viewBox="0 0 24 24" width="18" height="18" fill="none"
        stroke="currentColor" stroke-width="2.2"
        stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z"/>
      </svg>
    `;

    commentBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      openFeedbackModal({
        type: 'response',
        id,
        snippet: snippetForFeedback,
        job_id,
      });
    });

    actions.appendChild(copyBtn);
    actions.appendChild(commentBtn);

    head.appendChild(lbl);
    head.appendChild(actions);

    /* BODY */
    const body = document.createElement('div');
    body.className = 'message-text';
    body.textContent = '';

    /* FOOTER */
    const footer = document.createElement('div');
    footer.className = 'ab-footer';

    const selectBtn = document.createElement('button');
    selectBtn.type = 'button';
    selectBtn.className = 'ab-select-btn';
    selectBtn.textContent = 'Select this response';

    selectBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      selectPanel(panel, variant);
    });

    footer.appendChild(selectBtn);

    panel.appendChild(head);
    panel.appendChild(body);
    panel.appendChild(footer);

    panel.addEventListener('click', () => selectPanel(panel, variant));

    return {
      panel,
      body,
      setSnippet: (finalText) => {
        const s = String(finalText ?? '');
        snippetForFeedback = `[A/B ${variant}] ${s}`;
        latestTextForCopy = s;
      },
    };
  }

  const panelA = makePanel(meta.labelA || 'Response A', 'A', job_id_a);
  const panelB = makePanel(meta.labelB || 'Response B', 'B', job_id_b);

  wrap.appendChild(panelA.panel);
  wrap.appendChild(panelB.panel);

  inner.appendChild(avatar);
  inner.appendChild(wrap);
  row.appendChild(inner);

  els.messagesEl.appendChild(row);
  scrollChatToBottom();

  /* Typing animation */
  const aFinal = String(aText ?? '');
  const bFinal = String(bText ?? '');

  await typeIntoElement(panelA.body, aFinal, {
    cps: 60,
    chunkMin: 1,
    chunkMax: 4,
    maxTyped: 650,
  });
  panelA.setSnippet(aFinal);

  await typeIntoElement(panelB.body, bFinal, {
    cps: 60,
    chunkMin: 1,
    chunkMax: 4,
    maxTyped: 650,
  });
  panelB.setSnippet(bFinal);
}

/**
 * Appends a bot message row with copy and feedback buttons and returns
 * handles so the caller can type text into it and update the feedback
 * snippet once the final text is known.
 * @param {string=} initialText Initial body text (usually empty).
 * @param {string=} job_id Associated job id (defaults to JOB_ID_NONE).
 * @returns {{textEl: !Element, setSnippet: function(string): void,
 *     msgId: string}} The text element to type into, a snippet setter,
 *     and the generated message id.
 */
function appendBotTypingMessage(initialText = '', job_id = CFG.JOB_ID_NONE) {
  const row = document.createElement('div');
  row.className = 'message-row bot';

  const inner = document.createElement('div');
  inner.className = 'message-content';

  const avatar = document.createElement('div');
  avatar.className = 'message-avatar bot';
  setBotAvatar(avatar);

  const msgId = `bot_${++botMsgCounter}`;

  const contentWrap = document.createElement('div');
  contentWrap.style.display = 'flex';
  contentWrap.style.alignItems = 'flex-start';
  contentWrap.style.gap = '10px';
  contentWrap.style.width = '100%';

  const textEl = document.createElement('div');
  textEl.className = 'message-text';
  textEl.style.flex = '1';
  textEl.textContent = initialText;

  // attach job_id
  textEl.dataset.jobId = job_id;

  const actions = document.createElement('div');
  actions.className = 'msg-actions';

  // IMPORTANT: snippet should reflect FINAL text (not whatever it was initially)
  let snippetForFeedback = initialText;

  // Also keep the latest full text for copying
  let latestTextForCopy = initialText;

  // --- COPY button (new) ---
  const copyBtn = document.createElement('button');
  copyBtn.type = 'button';
  // reuse your existing button styling + tooltip behavior
  copyBtn.className = 'copy-btn comment-btn has-tooltip';
  copyBtn.dataset.tooltip = 'Copy to clipboard';
  copyBtn.setAttribute('aria-label', 'Copy response to clipboard');
  copyBtn.innerHTML = `
    <svg viewBox="0 0 24 24" width="18" height="18" fill="none"
      stroke="currentColor" stroke-width="2.2"
      stroke-linecap="round" stroke-linejoin="round">
      <rect x="9" y="9" width="13" height="13" rx="2"></rect>
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
    </svg>
  `;

  copyBtn.addEventListener('click', async (e) => {
    e.stopPropagation();
    try {
      const ok = await copyTextToClipboard(
        latestTextForCopy || textEl.textContent || '',
      );
      if (ok) bumpTooltip(copyBtn, 'Copied!');
      else bumpTooltip(copyBtn, 'Copy failed');
    } catch {
      bumpTooltip(copyBtn, 'Copy failed');
    }
  });

  // --- FEEDBACK button (existing) ---
  const commentBtn = document.createElement('button');
  commentBtn.type = 'button';
  commentBtn.className = 'comment-btn has-tooltip';
  commentBtn.dataset.tooltip = 'Add feedback';
  commentBtn.setAttribute('aria-label', 'Add feedback');
  commentBtn.innerHTML = `
    <svg viewBox="0 0 24 24" width="18" height="18" fill="none"
      stroke="currentColor" stroke-width="2.2"
      stroke-linecap="round" stroke-linejoin="round">
      <path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z"/>
    </svg>
  `;
  commentBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    openFeedbackModal({
      type: 'response',
      id: msgId,
      snippet: snippetForFeedback,
      job_id,
    });
  });

  // Put copy immediately LEFT of feedback
  actions.appendChild(copyBtn);
  actions.appendChild(commentBtn);
  contentWrap.appendChild(textEl);
  contentWrap.appendChild(actions);

  inner.appendChild(avatar);
  inner.appendChild(contentWrap);
  row.appendChild(inner);

  els.messagesEl.appendChild(row);
  scrollChatToBottom();

  return {
    textEl,
    setSnippet: (finalText) => {
      const s = String(finalText ?? '');
      snippetForFeedback = s;
      latestTextForCopy = s;
    },
    msgId,
  };
}

/* =========================================================
   11) REFERENCES
   ========================================================= */
/**
 * Turns bare http(s) URLs in an HTML string into anchor tags, while
 * conservatively avoiding URLs already inside tags/attributes.
 * @param {*} htmlStr HTML string to process (coerced to string).
 * @returns {string} HTML with bare URLs converted to links.
 */
function linkifyPlainUrls(htmlStr) {
  // Turn bare URLs into <a> links, but avoid touching existing tags too much.
  // This is intentionally conservative: it won't linkify inside existing attributes.
  const urlRe = /(^|[\s>])(https?:\/\/[^\s<]+)/g;
  return String(htmlStr || '').replace(urlRe, (m, prefix, url) => {
    return `${prefix}<a href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>`;
  });
}

/**
 * Normalizes a raw URL-ish string into an absolute https URL, handling
 * escaped slashes, protocol-relative URLs, and bare domains.
 * @param {*} raw Raw URL string (coerced to string).
 * @returns {string} An absolute https URL, or '' when not URL-like.
 */
function toHttpsUrl(raw) {
  let s = String(raw || '').trim();
  if (!s) return '';

  // If it's already a valid absolute URL, keep it
  if (/^https?:\/\//i.test(s)) return s;

  // Common escape sequences from JSON or stored strings
  s = s.replace(/\\\//g, '/'); // turns https:\/\/ into https://

  // Protocol-relative URLs
  if (s.startsWith('//')) return 'https:' + s;

  // "www.example.com/..." or "example.com/..."
  if (/^[a-z0-9.-]+\.[a-z]{2,}(\/|$)/i.test(s)) return 'https://' + s;

  return '';
}

/**
 * Picks the best link for a reference, preferring source_url, then url,
 * then a URL-looking source field.
 * @param {Object} ref A reference object.
 * @returns {string} The chosen link, or '' when none is available.
 */
function getRefLink(ref) {
  // Prefer new field names first
  const su = String(ref?.source_url || '').trim();
  if (su) return su;

  // Back-compat
  const u = String(ref?.url || '').trim();
  if (u) return u;

  const s = String(ref?.source || '').trim();
  if (looksLikeUrl(s)) return s;

  return '';
}

/**
 * Minimally sanitizes an HTML string: removes script/style/iframe and
 * similar elements, strips on* event handlers and javascript: URLs, and
 * forces safe target/rel on anchors. Not a bulletproof sanitizer.
 * @param {*} htmlStr HTML string to sanitize (coerced to string).
 * @returns {string} The sanitized HTML.
 */
function sanitizeBasicHtml(htmlStr) {
  // Minimal sanitizer (NOT bulletproof). Best practice is DOMPurify.
  // Removes scripts/styles/iframes and strips on* handlers.
  const tpl = document.createElement('template');
  tpl.innerHTML = String(htmlStr || '');

  // Remove dangerous elements
  tpl.content
    .querySelectorAll('script, style, iframe, object, embed, link, meta')
    .forEach((n) => n.remove());

  // Strip inline event handlers and javascript: URLs
  tpl.content.querySelectorAll('*').forEach((el) => {
    [...el.attributes].forEach((attr) => {
      const name = attr.name.toLowerCase();
      const val = String(attr.value || '')
        .trim()
        .toLowerCase();
      if (name.startsWith('on')) el.removeAttribute(attr.name);
      if ((name === 'href' || name === 'src') && val.startsWith('javascript:'))
        el.removeAttribute(attr.name);
    });

    // Force safe link behavior
    if (el.tagName === 'A') {
      el.setAttribute('target', '_blank');
      el.setAttribute('rel', 'noopener noreferrer');
    }
  });

  return tpl.innerHTML;
}

/**
 * Trims an element's text content to at most maxWords words by walking
 * its text nodes, appending an ellipsis at the cut point, and removing
 * any now-empty descendant elements. Preserves the HTML structure.
 * @param {!Element} containerEl Element whose text should be truncated.
 * @param {number} maxWords Maximum number of words to keep.
 * @returns {void}
 */
function truncateElementToWords(containerEl, maxWords) {
  // Walk text nodes and trim after maxWords, preserving HTML structure.
  const walker = document.createTreeWalker(containerEl, NodeFilter.SHOW_TEXT);
  let wordsUsed = 0;
  let node;

  const nodesToClearAfter = [];
  let trimming = false;

  while ((node = walker.nextNode())) {
    if (trimming) {
      nodesToClearAfter.push(node);
      continue;
    }

    const text = node.nodeValue || '';
    const parts = text.split(/\s+/).filter(Boolean);

    if (parts.length === 0) continue;

    if (wordsUsed + parts.length <= maxWords) {
      wordsUsed += parts.length;
      continue;
    }

    // Need to cut inside this node
    const remaining = maxWords - wordsUsed;
    const kept = parts.slice(0, Math.max(0, remaining)).join(' ');
    node.nodeValue = kept + '…';
    trimming = true;
  }

  // Remove all remaining text nodes content (and any elements that become empty)
  nodesToClearAfter.forEach((n) => {
    n.nodeValue = '';
  });

  // Cleanup: remove now-empty elements to avoid lots of blank tags
  containerEl.querySelectorAll('*').forEach((el) => {
    if (!el.textContent.trim() && !el.querySelector('img, br, hr')) {
      el.remove();
    }
  });
}

/**
 * Renders the references panel from a list of reference objects, building
 * one card per reference (subset link, optional dev-only scoring, title,
 * date, sanitized/truncated snippet, feedback button). Shows the empty
 * state when there are none.
 * @param {Array<Object>} refs Reference objects; capped at MAX_REFS.
 * @returns {void}
 */
function setReferences(refs) {
  const list = Array.isArray(refs) ? refs.slice(0, CFG.MAX_REFS) : [];
  els.referencesContainer.innerHTML = '';

  if (!list.length) {
    els.referencesCount.textContent = '0 items';
    els.referencesEmpty.style.display = 'block';
    return;
  }

  els.referencesEmpty.style.display = 'none';
  els.referencesCount.textContent =
    list.length + ' item' + (list.length === 1 ? '' : 's');

  list.forEach((ref, idx) => {
    if (idx === 0) {
      console.log('REF KEYS:', Object.keys(ref || {}));
      console.log('REF FULL:', ref);
    }

    const card = document.createElement('article');
    card.className = 'ref-card';

    const refId = `ref_${idx}_${Math.random().toString(16).slice(2)}`;

    const header = document.createElement('div');
    header.className = 'ref-header';

    const left = document.createElement('div');
    left.className = 'ref-left';

    const right = document.createElement('div');
    right.className = 'ref-right';

    // Show "Subset: {subset}" but keep the same link destination logic
    const subsetLabel =
      String(ref?.subset || '').trim() ||
      String(ref?.dataset || '').trim() ||
      'Unknown';

    // References scoring info only displayed when dev_mode is true
    var scoringLabel = null;
    var scoringLabelTxt = '';
    if (CFG.DEV_MODE) {
      scoringLabelTxt =
        ' BM25: ' +
        formatScore(ref?.score_bm25) +
        ', entity_score: ' +
        formatScore(ref?.entity_score) +
        ', fulltext_score: ' +
        formatScore(ref?.fulltext_score) +
        ', raw_score: ' +
        formatScore(ref?.raw_score);
      scoringLabel = document.createElement('div');
      scoringLabel.className = 'scoring-label';
      scoringLabel.textContent = scoringLabelTxt;
    } else {
    }

    const hrefRaw =
      String(ref?.source_url || '').trim() || String(ref?.source || '').trim();
    const href = toHttpsUrl(hrefRaw);

    /*
    if (idx === 0) {
      console.log(
        'source_url=',
        ref?.source_url,
        'source=',
        ref?.source,
        'subset=',
        ref?.subset,
      );
    }
    console.log('subset hrefRaw=', hrefRaw, 'href=', href);
    */

    const foundEl = document.createElement(href ? 'a' : 'span');
    foundEl.className = 'ref-foundon';
    foundEl.textContent = `Subset: ${subsetLabel}`;

    if (href) {
      foundEl.href = href;
      foundEl.target = '_blank';
      foundEl.rel = 'noopener noreferrer';
      foundEl.addEventListener('click', (e) => e.stopPropagation());
    }

    left.appendChild(foundEl);
    //console.log('foundEl tag=', href ? 'a' : 'span', 'hrefRaw=', hrefRaw, 'href=', href);

    // Scoring only displayed in dev mode
    if (scoringLabel !== null) {
      left.appendChild(scoringLabel);
    }

    const cbtn = document.createElement('button');
    cbtn.type = 'button';
    cbtn.className = 'comment-btn';
    cbtn.dataset.tooltip = 'Add feedback';
    cbtn.setAttribute('aria-label', 'Add feedback');
    cbtn.innerHTML = `
      <svg viewBox="0 0 24 24" width="16" height="16" fill="none"
        stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z"/>
      </svg>
    `;
    cbtn.addEventListener('click', (e) => {
      e.stopPropagation();
      const title = String(ref?.title || '').trim();
      const snip = String(ref?.snippet || ref?.text || '').trim();
      openFeedbackModal({
        type: 'reference',
        id: refId,
        snippet: (title ? title + ' — ' : '') + snip,
        job_id: 'none',
      });
    });

    right.appendChild(cbtn);

    header.appendChild(left);
    header.appendChild(right);

    const titleText = String(ref?.title || '').trim();
    if (titleText) {
      const t = document.createElement('div');
      t.className = 'ref-title';
      t.textContent = titleText;
      header.appendChild(t);
    }

    const metaBits = [];
    const date = String(ref?.date || '').trim();
    if (date) metaBits.push(date);

    if (metaBits.length) {
      const meta = document.createElement('div');
      meta.className = 'ref-meta';
      meta.textContent = metaBits.join(' • ');
      header.appendChild(meta);
    }

    const snippetEl = document.createElement('div');
    snippetEl.className = 'ref-snippet';

    let rawText = String(
      ref?.snippet ?? ref?.text ?? ref?.content ?? '',
    ).trim();
    let rawHtml = String(ref?.snippet_html ?? '').trim();

    if (rawText) {
      rawText = cleanRagArtifacts(rawText);
      const mdHtml = renderMiniMarkdown(rawText);
      snippetEl.innerHTML = sanitizeBasicHtml(mdHtml);
    } else if (rawHtml) {
      rawHtml = cleanRagArtifacts(rawHtml);
      snippetEl.innerHTML = sanitizeBasicHtml(rawHtml);
    } else {
      snippetEl.textContent = '';
    }

    truncateElementToWords(snippetEl, 600);

    card.appendChild(header);
    card.appendChild(snippetEl);
    els.referencesContainer.appendChild(card);
  });
}

/* =========================================================
   12) CONVERSATION HISTORY (client-side)
   ========================================================= */
/**
 * Appends a conversation turn to the in-memory history, dropping the
 * oldest turn once MAX_CONVO_TURNS is exceeded.
 * @param {string} userText The user's message.
 * @param {string} assistantText The assistant's reply.
 * @returns {void}
 */
function pushTurn(userText, assistantText) {
  convoTurns.push({ user: userText, assistant: assistantText });
  if (convoTurns.length > CFG.MAX_CONVO_TURNS) convoTurns.shift();
}

/**
 * Returns the most recent N conversation turns for use as request context.
 * @param {(number|string)} n Requested number of turns; clamped to
 *     [0, MAX_CONTEXT_TURNS].
 * @returns {!Array<{user: string, assistant: string}>} The recent turns.
 */
function getContextTurns(n) {
  const turns = clampInt(
    n,
    0,
    CFG.MAX_CONTEXT_TURNS,
    CFG.DEFAULT_HISTORY_TURNS,
  );
  return convoTurns.slice(-turns);
}

/* =========================================================
   13) SAVED CONVERSATIONS (localStorage)
   ========================================================= */
/**
 * Loads the list of saved conversations from localStorage.
 * @returns {!Array<Object>} The saved conversation records, or [].
 */
function loadSavedConversations() {
  const raw = localStorage.getItem(CFG.LS_SAVED_CONVOS);
  const parsed = safeJsonParse(raw || '[]', []);
  return Array.isArray(parsed) ? parsed : [];
}

/**
 * Persists the list of saved conversations to localStorage.
 * @param {!Array<Object>} list The conversation records to store.
 * @returns {void}
 */
function saveSavedConversations(list) {
  try {
    localStorage.setItem(CFG.LS_SAVED_CONVOS, JSON.stringify(list));
  } catch (_) {}
}

/**
 * Derives a short conversation title from the first user message.
 * @param {Array<{user: string, assistant: string}>} turns Conversation turns.
 * @returns {string} A truncated title string.
 */
function makeConversationTitle(turns) {
  const first = turns?.find((t) => t?.user)?.user || 'Conversation';
  const t = String(first).trim().replace(/\s+/g, ' ');
  return t.length > 42 ? t.slice(0, 42) + '…' : t;
}

/**
 * Derives a short preview string from the last turn of a conversation.
 * @param {Array<{user: string, assistant: string}>} turns Conversation turns.
 * @returns {string} A truncated preview string, or '' when empty.
 */
function makeConversationPreview(turns) {
  if (!Array.isArray(turns) || turns.length === 0) return '';
  const last = turns[turns.length - 1];
  const txt = String(last.assistant || last.user || '')
    .trim()
    .replace(/\s+/g, ' ');
  return txt.length > 60 ? txt.slice(0, 60) + '…' : txt;
}

/**
 * Renders the "recent conversations" sidebar list from localStorage,
 * sorted newest-first, with click-to-load and delete buttons.
 * @returns {void}
 */
function renderRecentList() {
  const saved = loadSavedConversations();
  els.recentList.innerHTML = '';

  if (!saved.length) {
    const el = document.createElement('div');
    el.className = 'recent-item muted';
    el.textContent = 'No saved conversations yet.';
    els.recentList.appendChild(el);
    return;
  }

  saved
    .slice()
    .sort((a, b) => (b.savedAt || 0) - (a.savedAt || 0))
    .forEach((item) => {
      const row = document.createElement('div');
      row.className = 'recent-preview';

      const content = document.createElement('div');
      content.className = 'recent-preview-content';

      const title = document.createElement('div');
      title.className = 'recent-preview-title';
      title.textContent = item.title || 'Conversation';

      const sub = document.createElement('div');
      sub.className = 'recent-preview-sub';
      sub.textContent = item.preview || '';

      content.appendChild(title);
      content.appendChild(sub);

      const del = document.createElement('button');
      del.type = 'button';
      del.className = 'recent-delete has-tooltip';
      del.dataset.tooltip = 'Delete conversation';
      del.setAttribute('aria-label', 'Delete conversation');
      del.textContent = '×';

      del.addEventListener('click', (e) => {
        e.stopPropagation();
        deleteConversationById(item.id);
      });

      row.addEventListener('click', () => loadConversationById(item.id));

      row.appendChild(content);
      row.appendChild(del);

      els.recentList.appendChild(row);
    });
}

/**
 * Saves the current in-memory conversation to localStorage and refreshes
 * the recent list. No-op (with a status message) when there is nothing
 * to save.
 * @returns {void}
 */
function saveCurrentConversation() {
  if (!Array.isArray(convoTurns) || convoTurns.length === 0) {
    pushStatusMessage('Nothing to save yet.');
    return;
  }

  const saved = loadSavedConversations();
  const item = {
    id: nowId('convo'),
    savedAt: Date.now(),
    title: makeConversationTitle(convoTurns),
    preview: makeConversationPreview(convoTurns),
    turns: convoTurns.slice(),
  };

  saved.unshift(item);
  if (saved.length > CFG.MAX_SAVED_CONVOS) saved.length = CFG.MAX_SAVED_CONVOS;

  saveSavedConversations(saved);
  renderRecentList();
  pushStatusMessage(`Saved conversation: "${item.title}"`);
}

/**
 * Loads a saved conversation by id into the chat view, replacing the
 * current messages and in-memory turns.
 * @param {string} id The saved conversation's id.
 * @returns {void}
 */
function loadConversationById(id) {
  const saved = loadSavedConversations();
  const item = saved.find((x) => x.id === id);
  if (!item) return;

  els.messagesEl.innerHTML = '';
  convoTurns.length = 0;
  if (els.welcomeMessage) els.welcomeMessage.style.display = 'none';

  (item.turns || []).forEach((t) => {
    if (t.user) appendMessage(t.user, 'user');
    if (t.assistant) appendMessage(t.assistant, 'bot');
    convoTurns.push({ user: t.user || '', assistant: t.assistant || '' });
  });

  pushStatusMessage(`Loaded conversation: "${item.title}"`);
}

/**
 * Prompts for confirmation, then deletes a saved conversation by id and
 * refreshes the recent list.
 * @param {string} id The saved conversation's id.
 * @returns {Promise<void>} Resolves when the deletion flow completes.
 */
async function deleteConversationById(id) {
  const ok = await modalConfirm({
    title: 'Delete saved conversation?',
    message:
      'This will remove the saved conversation from this browser. This cannot be undone.',
    confirmText: 'Delete',
    cancelText: 'Cancel',
    danger: true,
  });
  if (!ok) return;

  const saved = loadSavedConversations().filter((c) => c.id !== id);
  saveSavedConversations(saved);
  renderRecentList();
  pushStatusMessage('Conversation deleted.');
}

/* =========================================================
   14) INPUT AREA
   ========================================================= */
/**
 * Resizes the chat input textarea to fit its content, up to
 * TEXTAREA_MAX_HEIGHT, toggling vertical scrolling past that height.
 * @returns {void}
 */
function autoResizeTextarea() {
  if (!els.chatInput) return;
  els.chatInput.style.height = 'auto';
  els.chatInput.style.height =
    Math.min(els.chatInput.scrollHeight, CFG.TEXTAREA_MAX_HEIGHT) + 'px';
  els.chatInput.style.overflowY =
    els.chatInput.scrollHeight > CFG.TEXTAREA_MAX_HEIGHT ? 'auto' : 'hidden';
}

/**
 * Prompts for confirmation, then clears the current chat view, in-memory
 * turns, and references. Saved conversations are unaffected.
 * @returns {Promise<void>} Resolves when the clear flow completes.
 */
async function clearCurrentChat() {
  const hasMessages =
    (convoTurns && convoTurns.length) ||
    (els.messagesEl && els.messagesEl.children.length);
  if (!hasMessages) {
    await modalAlert({
      title: 'Nothing to clear',
      message: 'There are no messages in the current chat yet.',
    });
    return;
  }

  const ok = await modalConfirm({
    title: 'Clear current chat?',
    message:
      'This clears the current chat on this page. Saved chats are not affected.',
    confirmText: 'Clear',
    cancelText: 'Cancel',
    danger: true,
  });
  if (!ok) return;

  els.messagesEl.innerHTML = '';
  convoTurns.length = 0;
  if (els.welcomeMessage) els.welcomeMessage.style.display = '';
  setReferences([]);
  pushStatusMessage('Current chat cleared.');
}

/* =========================================================
   15) FLASK API HELPERS (session cookie enabled)
   ========================================================= */
/**
 * Sends a JSON POST request to a Flask endpoint with the session cookie,
 * parsing the JSON response. On a non-OK status it throws an Error
 * augmented with status, data, and job_id fields.
 * @param {string} url The endpoint URL.
 * @param {Object=} payload The request body object.
 * @returns {Promise<Object>} The parsed JSON response.
 */
async function apiPost(url, payload) {
  if (CFG.DEV_MODE) {
    console.log('API POST to url: ' + url);
  }

  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'same-origin', // required for Flask session cookie
    body: JSON.stringify(payload || {}),
  });

  let data = {};
  try {
    data = await res.json();
  } catch (_) {}

  //console.log("GOT res")

  if (!res.ok) {
    const msg =
      data?.message || data?.error || `Request failed (${res.status})`;
    const err = new Error(msg);
    err.status = res.status;
    err.data = data;

    // add job id to error for processing if present
    err.job_id = data?.job_id ?? null;

    throw err;
  }
  return data;
}

/**
 * Sends a GET request to a Flask endpoint with the session cookie and
 * parses the JSON response, throwing an Error on a non-OK status.
 * @param {string} url The endpoint URL.
 * @returns {Promise<Object>} The parsed JSON response.
 */
async function apiGet(url) {
  const res = await fetch(url, { method: 'GET', credentials: 'same-origin' });
  let data = {};
  try {
    data = await res.json();
  } catch (_) {}
  if (!res.ok)
    throw new Error(data?.message || `Request failed (${res.status})`);
  return data;
}

/**
 * POSTs a search request to /api/search, logging the raw/parsed response.
 * Treats both a non-OK HTTP status and an `ok:false` body as errors.
 * @param {!Object} payload The search request body.
 * @returns {Promise<Object>} The parsed JSON response.
 */
async function apiSearch(payload) {
  const res = await fetch('/api/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'same-origin',
    body: JSON.stringify(payload),
  });

  const text = await res.text();

  console.log('[/api/search] status:', res.status);
  console.log('[/api/search] raw body:', text.slice(0, 2000));

  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch (_) {}

  console.log('[/api/search] parsed keys:', Object.keys(data || {}));
  console.log('[/api/search] parsed data:', data);

  // ALSO treat ok:false as an error even if HTTP 200
  if (!res.ok || data.ok === false) {
    const msg = data.error || data.message || text || `HTTP ${res.status}`;
    throw new Error(`apiSearch failed: ${res.status} — ${msg}`);
  }

  return data;
}

/**
 * POSTs a chat request to /api/chat.
 * @param {!Object} payload The chat request body.
 * @returns {Promise<Object>} The parsed JSON response.
 */
async function apiChat(payload) {
  return apiPost(CFG.API.CHAT, payload);
}

/**
 * Streaming-capable model adapters. The chat send path routes these through
 * /api/chat/stream (Server-Sent Events) for a live token-by-token reply;
 * everything else uses the queued /api/chat + pollJob path unchanged. Keep in
 * sync with the adapters that implement generate_stream() in model_adapters.py
 * and with the dev-mode dropdown options.
 */
const STREAMING_MODEL_TYPES = new Set(['deepinfra_stream', 'vllm']);

/**
 * POSTs a chat turn to /api/chat/stream and consumes the Server-Sent Events
 * response, dispatching each frame to the supplied handlers. Pre-stream
 * failures (validation 400, rate-limit 429, gate 403) arrive as ordinary JSON
 * and are thrown as Errors (with `.status`) so the caller's catch can handle
 * them like the non-streaming path.
 *
 * @param {!Object} payload The chat request body.
 * @param {{onChunk?:function(string):void, onDone?:function(Object):void,
 *          onQueued?:function(Object):void, onError?:function(Object):void}} h
 *   Event handlers.
 * @returns {Promise<void>}
 */
async function apiChatStream(payload, h = {}) {
  const { onChunk, onDone, onQueued, onError } = h;
  const res = await fetch(CFG.API.CHAT_STREAM, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    credentials: 'same-origin',
    body: JSON.stringify(payload),
  });

  const ct = (res.headers.get('Content-Type') || '').toLowerCase();
  if (!ct.includes('text/event-stream')) {
    // Pre-stream failure (validation / rate limit / gate) — JSON, not SSE.
    let data = {};
    try { data = await res.json(); } catch (_) {}
    if (!res.ok) {
      const e = new Error(data?.error || data?.message || `HTTP ${res.status}`);
      e.status = res.status;
      throw e;
    }
    onError?.({ error: data?.error || 'Unexpected non-streaming response' });
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';

  const dispatch = (rawEvent) => {
    let name = 'message';
    let dataStr = '';
    for (const line of rawEvent.split('\n')) {
      if (line.startsWith('event:')) name = line.slice(6).trim();
      else if (line.startsWith('data:')) dataStr += line.slice(5).trim();
    }
    let data = {};
    if (dataStr) {
      try { data = JSON.parse(dataStr); } catch (_) { data = { text: dataStr }; }
    }
    if (name === 'chunk') onChunk?.(data.text || '');
    else if (name === 'done') onDone?.(data);
    else if (name === 'queued') onQueued?.(data);
    else if (name === 'error') onError?.(data);
  };

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    // SSE frames are separated by a blank line.
    let idx;
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      const rawEvent = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      if (rawEvent.trim()) dispatch(rawEvent);
    }
  }
  if (buf.trim()) dispatch(buf);  // flush a trailing frame, if any
}

/**
 * Builds a throttled, markdown-rendering sink bound to one bubble element.
 * push() appends a chunk and re-renders the accumulated text as formatted
 * Markdown (coalesced so a fast token stream doesn't thrash the parser);
 * finalize() does a last render (optionally swapping in a server-cleaned
 * final reply) and stops further updates. `.text` exposes the accumulator.
 *
 * @param {!Element} el The bubble's text element.
 * @returns {{push:function(string):void, finalize:function(string=):void, text:string}}
 */
function makeStreamRenderer(el) {
  let acc = '';
  let finalized = false;
  let scheduled = false;
  let lastRender = 0;
  const MIN_MS = 60;

  if (el) el.classList.add('md-rendered');

  const render = () => {
    if (!el) return;
    // Follow the stream only if the user is already near the bottom, so
    // scrolling up to read isn't yanked back on every token.
    const wrap = els.messagesEl;
    const nearBottom =
      !wrap || (wrap.scrollHeight - wrap.scrollTop - wrap.clientHeight) < 80;
    el.innerHTML = renderMiniMarkdown(acc, { blocks: true });
    if (nearBottom && typeof scrollChatToBottom === 'function') scrollChatToBottom();
  };

  return {
    push(chunk) {
      if (finalized) return;
      acc += String(chunk ?? '');
      const now = Date.now();
      if (now - lastRender >= MIN_MS) {
        lastRender = now;
        render();
      } else if (!scheduled) {
        scheduled = true;
        setTimeout(() => {
          scheduled = false;
          lastRender = Date.now();
          render();
        }, MIN_MS);
      }
    },
    finalize(finalText) {
      finalized = true;
      if (typeof finalText === 'string' && finalText) acc = finalText;
      if (el) {
        el.classList.add('md-rendered');
        el.innerHTML = renderMiniMarkdown(acc, { blocks: true });
      }
    },
    get text() {
      return acc;
    },
  };
}

/**
 * Runs a streamed chat turn against /api/chat/stream, rendering tokens into
 * `chatBotUI` live as Markdown. Returns true when the turn was fully handled
 * here (streamed to a done reply, or an error was shown). Returns false when
 * the server queued the job instead (model not ready); in that case the
 * queued event is stashed on `payload.__queued` so the caller resumes the
 * normal /api/job polling path. Throws (propagating to the caller's catch)
 * on pre-stream failures like 403/429/network, matching the non-stream path.
 *
 * @param {!Object} payload The chat request body.
 * @param {string} text The user's message (for pushTurn).
 * @param {!Object} chatBotUI The placeholder bubble ({textEl, setSnippet}).
 * @param {string} userId The current user id.
 * @returns {Promise<boolean>} true if handled here; false if queued.
 */
async function runChatStream(payload, text, chatBotUI, userId) {
  const streamRenderer = makeStreamRenderer(chatBotUI.textEl);
  let firstChunk = true;
  let doneData = null;
  let queuedData = null;
  let errorData = null;

  await apiChatStream(payload, {
    onChunk: (piece) => {
      if (!piece) return;
      if (firstChunk) {
        firstChunk = false;
        // Drop the "Asking…" waiting copy the instant real content begins.
        if (chatBotUI?.textEl) chatBotUI.textEl.textContent = '';
      }
      streamRenderer.push(piece);
    },
    onDone: (d) => { doneData = d; },
    onQueued: (d) => { queuedData = d; },
    onError: (d) => { errorData = d; },
  });

  // Model not ready: server queued the job. Restore the plain-text bubble and
  // hand back to the caller's polling path via payload.__queued.
  if (queuedData) {
    if (chatBotUI?.textEl) chatBotUI.textEl.classList.remove('md-rendered');
    payload.__queued = queuedData;
    return false;
  }

  if (errorData) {
    const detail = errorData.detail || errorData.error || 'Chat failed.';
    if (chatBotUI?.textEl) {
      chatBotUI.textEl.classList.remove('md-rendered');
      chatBotUI.textEl.textContent = detail;
    }
    pushStatusMessage(detail);
    setReferences([]);
    return true;
  }

  // Success: finalize with the server-cleaned reply (falls back to the raw
  // streamed text if the done event was empty).
  const d = doneData || {};
  const finalReply = d.reply || streamRenderer.text || '(no reply)';
  streamRenderer.finalize(finalReply);
  chatBotUI.setSnippet?.(finalReply);
  pushTurn(text, finalReply);
  setReferences(Array.isArray(d.references) ? d.references : []);
  return true;
}

/**
 * POSTs an A/B request to /api/ab.
 * @param {!Object} payload The A/B request body.
 * @returns {Promise<Object>} The parsed JSON response.
 */
async function apiAB(payload) {
  return apiPost(CFG.API.AB, payload);
}

/**
 * POSTs a password to /api/unlock to attempt to unlock chat access.
 * @param {string} password The candidate password.
 * @returns {Promise<Object>} The parsed JSON response.
 */
async function apiUnlock(password) {
  return apiPost(CFG.API.UNLOCK, { password });
}

/**
 * GETs /api/access to check whether the current session is unlocked.
 * @returns {Promise<Object>} The parsed JSON response.
 */
async function apiAccess() {
  return apiGet(CFG.API.ACCESS);
}

/**
 * Fetches a URL and parses its JSON response, aborting after a timeout.
 * Throws an Error on a non-OK status.
 * @param {string} url The URL to fetch.
 * @param {!Object=} opts Fetch options merged with the abort signal.
 * @param {number=} timeoutMs Timeout in milliseconds before aborting.
 * @returns {Promise<Object>} The parsed JSON response.
 */
async function fetchJsonWithTimeout(url, opts = {}, timeoutMs = 4000) {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(url, { ...opts, signal: controller.signal });
    let data = {};
    try {
      data = await res.json();
    } catch (_) {}
    if (!res.ok) {
      throw new Error(data?.error || data?.message || `HTTP ${res.status}`);
    }
    return data;
  } finally {
    clearTimeout(t);
  }
}

/**
 * GETs /api/feedback.
 * @returns {Promise<Object>} The parsed JSON response.
 */
async function apiFeedback() {
  return apiGet(CFG.API.FEEDBACK);
}

/**
 * POSTs an email-response request, attaching an email address to a
 * queued job so the result can be emailed when ready.
 * @param {string} job_id The queued job's id.
 * @param {string} user_id The requesting user's id.
 * @param {string} email The email address to deliver the response to.
 * @returns {Promise<Object>} The parsed JSON response.
 */
async function apiEmailResponse(job_id, user_id, email) {
  return apiPost(CFG.API.EMAIL_RESPONSE, { job_id, user_id, email });
}

/**
 * POSTs a request to cancel a previously-attached email response.
 * @param {string} job_id The queued job's id.
 * @param {string} user_id The requesting user's id.
 * @returns {Promise<Object>} The parsed JSON response.
 */
async function apiEmailResponseCancel(job_id, user_id) {
  return apiPost(CFG.API.EMAIL_RESPONSE_CANCEL, { job_id, user_id });
}

/**
 * POSTs a status request to /api/status with a generous timeout.
 * @param {{health?: boolean, model_type?: ?string}=} options Whether to
 *     request a health check and an optional model type to query.
 * @returns {Promise<Object>} The parsed JSON status response.
 */
async function apiStatus({ health = true, model_type = null } = {}) {
  const body = { health: String(health) };

  if (model_type) {
    body.model_type = model_type;
  }

  return fetchJsonWithTimeout(
    CFG.API.STATUS,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify(body),
    },
    30000,
  );
}

/**
 * POSTs a queue-status request to /api/queue for the current user.
 * @returns {Promise<Object>} The parsed JSON queue response.
 */
async function apiQueue() {
  return apiPost(CFG.API.QUEUE, {
    user_id: getUserId(),
  });
}

/* ---------------------------------------------------------
   Per-job status polling
   GET /api/job/<id>?user_id=… — see app.py:api_job_status
   --------------------------------------------------------- */

async function apiJobStatus(job_id, user_id) {
  const url = `${CFG.API.JOB}/${encodeURIComponent(job_id)}`
    + `?user_id=${encodeURIComponent(user_id)}`;
  return fetchJsonWithTimeout(url, {
    method: 'GET',
    credentials: 'same-origin',
  }, 15000);
}

/**
 * Poll /api/job/<id> until the job reaches a terminal state ('done' or
 * 'failed'), the hard timeout fires, or the network gives up.
 *
 * Returns one of:
 *   { status: 'done',   reply, references }
 *   { status: 'failed', error, detail }
 *
 * Cleans up activeJobs + sessionStorage on terminal state.
 */
async function pollJob(job_id, user_id, placeholder, initialDelayMs) {
  // Server can advise the initial poll cadence via /api/chat's
  // `poll_interval_ms` field; fall back to the client default.
  let delay = (typeof initialDelayMs === 'number' && initialDelayMs > 0)
    ? initialDelayMs
    : CFG.JOB_POLL_INITIAL_MS;
  const startedAt = Date.now();
  let netFails = 0;
  let lastSurfacedHint = '';

  function setHint(text) {
    if (!placeholder?.textEl) return;
    if (text === lastSurfacedHint) return;
    lastSurfacedHint = text;
    // Only update if the bubble still has the waiting copy — once we
    // start typing the real reply we don't want to clobber it.
    if (!placeholder.textEl.dataset.replyStarted) {
      placeholder.textEl.textContent = text;
    }
  }

  try {
    while (Date.now() - startedAt < CFG.JOB_POLL_TOTAL_MS) {
      try {
        const data = await apiJobStatus(job_id, user_id);
        netFails = 0;

        if (data.status === 'done') {
          return {
            status: 'done',
            reply: data.reply || '',
            references: Array.isArray(data.references) ? data.references : [],
          };
        }
        if (data.status === 'failed') {
          return {
            status: 'failed',
            error: data.error || 'Chat failed',
            detail: data.detail || '',
          };
        }
        if (data.status === 'processing') {
          setHint('Working on your answer…');
        } else if (data.status === 'queued') {
          const pos = data.queue_position;
          setHint(pos ? `Queued (position ${pos})…` : 'Queued…');
        }
      } catch (e) {
        netFails++;
        if (netFails === CFG.JOB_POLL_NET_FAIL_NOTIFY) {
          pushStatusMessageDedup('Lost connection, retrying…', 'poll-net');
        }
        // 404 from the GET typically means the job_id doesn't belong to
        // this user (or sessionStorage held a stale id). Bail.
        if (/HTTP 404/.test(String(e?.message || ''))) {
          return { status: 'failed', error: 'Job not found', detail: '' };
        }
      }

      await new Promise(r => setTimeout(r, delay));
      delay = Math.min(delay + 500, CFG.JOB_POLL_MAX_MS);
    }
    return {
      status: 'failed',
      error: 'Timed out',
      detail: 'No response from the model after 10 minutes.',
    };
  } finally {
    activeJobs.delete(job_id);
    setActiveJobIdHint();
    clearActiveJob(job_id);
    if (placeholder?.textEl) {
      placeholder.textEl.dataset.replyStarted = '1';
    }
  }
}

/* ---------------------------------------------------------
   SessionStorage resume — recover in-flight job on reload
   --------------------------------------------------------- */

function persistActiveJob(job_id, user_id) {
  try {
    sessionStorage.setItem(
      CFG.ACTIVE_JOB_SS_KEY,
      JSON.stringify({ job_id, user_id, at: Date.now() })
    );
  } catch (e) {
    // sessionStorage can throw (Safari private mode, quota). Non-fatal —
    // user just loses reload-resume for this job.
    console.warn('persistActiveJob: sessionStorage write failed', e);
  }
}

function clearActiveJob(job_id) {
  try {
    const raw = sessionStorage.getItem(CFG.ACTIVE_JOB_SS_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    // Only clear if it matches — a newer job may have already overwritten it.
    if (!job_id || parsed?.job_id === job_id) {
      sessionStorage.removeItem(CFG.ACTIVE_JOB_SS_KEY);
    }
  } catch (e) {
    sessionStorage.removeItem(CFG.ACTIVE_JOB_SS_KEY);
  }
}

/**
 * On page init, if sessionStorage holds a recent unfinished job_id,
 * recreate a placeholder bubble and resume polling. Called from the
 * init flow at the bottom of this file.
 */
async function resumeActiveJobIfAny() {
  let parsed = null;
  try {
    const raw = sessionStorage.getItem(CFG.ACTIVE_JOB_SS_KEY);
    if (!raw) return;
    parsed = JSON.parse(raw);
  } catch (e) {
    sessionStorage.removeItem(CFG.ACTIVE_JOB_SS_KEY);
    return;
  }
  if (!parsed?.job_id || !parsed?.user_id) {
    sessionStorage.removeItem(CFG.ACTIVE_JOB_SS_KEY);
    return;
  }
  const age = Date.now() - (parsed.at || 0);
  if (age > CFG.ACTIVE_JOB_RESUME_MAX_AGE_MS) {
    sessionStorage.removeItem(CFG.ACTIVE_JOB_SS_KEY);
    return;
  }

  console.log(`[resume] picking up job_id=${parsed.job_id} (age=${Math.round(age/1000)}s)`);
  const placeholder = appendBotTypingMessage('', parsed.job_id);
  try {
    await typeIntoElement(
      placeholder.textEl,
      'Resuming previous request… (this tab was reloaded mid-response)',
      { cps: 70, chunkMin: 1, chunkMax: 4, maxTyped: 800 }
    );
  } catch (_) {}

  activeJobs.set(parsed.job_id, { placeholder, startedAt: Date.now() });
  setActiveJobIdHint();

  const result = await pollJob(parsed.job_id, parsed.user_id, placeholder);
  if (result.status === 'done') {
    const finalReply = result.reply || '(no reply)';
    if (placeholder?.textEl) placeholder.textEl.textContent = '';
    await typeIntoElement(placeholder.textEl, finalReply, {
      cps: 60, chunkMin: 1, chunkMax: 4, maxTyped: 800,
    });
    placeholder.setSnippet?.(finalReply);
    setReferences(Array.isArray(result.references) ? result.references : []);
  } else {
    const detail = result.detail || result.error || 'Chat failed.';
    if (placeholder?.textEl) placeholder.textEl.textContent = '';
    await typeIntoElement(placeholder.textEl, detail, {
      cps: 80, chunkMin: 1, chunkMax: 4, maxTyped: 800,
    });
    pushStatusMessage(detail);
  }
}

/**
 * Decides which model backend to use for the next submit. Refreshes the
 * Spark health check when the cached value is stale, and routes to
 * DeepInfra when Spark is unavailable or the health check fails.
 * @returns {Promise<string>} 'spark' or 'deepinfra'.
 */
async function chooseModelTypeForSubmit() {
  // If the user explicitly picked a non-default model from the DEV_MODE
  // dropdown, honor their choice. The Spark/DeepInfra health-routing
  // logic below only applies when the user wants the default Spark
  // routing — otherwise their selection of e.g. "sim" or "deepinfra"
  // would be silently overridden and they'd see the wrong backend's
  // errors (the symptom that surfaced this bug: selecting "Sim Adapter
  // (testing)" still produced a Cloudflare 524 from the Spark origin).
  //
  // In non-DEV_MODE this is a no-op: loadToolState() force-pins
  // toolState.modelType = 'spark', so the equality check below always
  // skips this branch and we run the normal Spark/DeepInfra routing.
  const userPicked = toolState.modelType;
  if (userPicked && userPicked !== PRIMARY_MODEL_TYPE) {
    console.log(`[model] honoring explicit selection: ${userPicked}`);
    return userPicked;
  }

  const now = Date.now();
  const healthIsStale =
    lastKnownSparkReady === null ||
    now - lastSparkHealthCheckAt > SPARK_HEALTH_MAX_AGE_MS;

  if (healthIsStale) {
    try {
      // Must pass model_type explicitly: without it the server reports the
      // readiness of its global default backend (MODEL_ADAPTER env var, often
      // "hf"), NOT Spark — which would make this routing decision meaningless.
      const data = await apiStatus({ health: true, model_type: PRIMARY_MODEL_TYPE });

      if (typeof data?.model_ready === 'boolean') {
        lastKnownSparkReady = data.model_ready;
        lastSparkHealthCheckAt = now;
      }
    } catch (err) {
      console.warn(
        '[model fallback] Primary model health check failed:',
        err?.message || err,
      );

      // If the Spark health check itself fails, don't block the user.
      // Route to DeepInfra.
      lastKnownSparkReady = false;
      lastSparkHealthCheckAt = now;
    }
  }

  return lastKnownSparkReady === false ? FALLBACK_MODEL_TYPE : PRIMARY_MODEL_TYPE;
}

/* =========================================================
   16) CHAT HANDLING (submit + render)
   ========================================================= */
/**
 * Returns a stable per-browser user id, generating and persisting one in
 * localStorage on first use.
 * @returns {string} The user id.
 */
function getUserId() {
  const KEY = 'sot_user_id';
  let id = localStorage.getItem(KEY);
  if (!id) {
    // Stable per-browser ID (good enough for dev + non-auth chat)
    id = crypto?.randomUUID
      ? crypto.randomUUID()
      : `u_${Date.now()}_${Math.random().toString(16).slice(2)}`;
    localStorage.setItem(KEY, id);
  }
  return id;
}

/**
 * Handles submission of the chat form. Reads the input, appends the user
 * message, chooses the model backend, optionally runs the proactive
 * email opt-in, then dispatches to the search, A/B, or chat flow. Renders
 * the response (with a typing animation), updates references, and handles
 * queued (503), access-revoked (403), and generic error cases.
 * @param {!Event} e The form submit event.
 * @returns {Promise<void>} Resolves when the submission flow completes.
 */
async function handleChatSubmit(e) {
  e.preventDefault();
  const text = (els.chatInput.value || '').trim();
  if (!text) return;

  const userId = getUserId();

  if (els.welcomeMessage) els.welcomeMessage.style.display = 'none';

  appendMessage(text, 'user');
  els.chatInput.value = '';
  autoResizeTextarea();

  const contextTurns = getContextTurns(toolState.historyTurns);

  // Define model type. Prefer Spark, but fall back to DeepInfra if Spark is unavailable.
  const model_type = await chooseModelTypeForSubmit();

  if (model_type === FALLBACK_MODEL_TYPE) {
    pushStatusMessage('Spark unavailable; using DeepInfra fallback.');
  }

  // -------- Proactive email pre-submit check --------
  // If the queue is busy enough and the server has reported that email
  // delivery is offered, prompt the user up-front: "queue is busy — want
  // an email instead of waiting?" If they pick email, we set
  // force_queue=true for THIS submit so the request lands in the queue
  // path and produces a job_id we can attach the email to. If they pick
  // wait, we proceed normally and remember the decision so the
  // post-queue modal doesn't re-prompt them.
  let proactiveForceQueue = false;
  if (
    CFG.PROACTIVE_EMAIL_THRESHOLD > 0 &&
    lastKnownEmailOffer &&
    lastKnownQueueDepth >= CFG.PROACTIVE_EMAIL_THRESHOLD &&
    toolState.mode === 'chat' &&
    !preSubmitEmail // not already mid-flight
  ) {
    const result = await offerEmailPreSubmit({
      queue_position: lastKnownQueueDepth + 1,
      queue_reason: 'queue_busy',
    });

    if (result?.decision === 'emailed' && result.email) {
      preSubmitEmail = result.email;
      proactiveForceQueue = true;
    } else {
      // User chose to wait (or dismissed). Mark the decision so the
      // 503 catch — if this submit still ends up queueing — doesn't
      // open a second modal asking the same question.
      preSubmitDeclined = true;
    }
  }

  // Base payload (chat / ab). Search uses query instead of message.
  const payload = {
    user_id: userId,
    message: text,
    mode: toolState.mode,
    history_turns: toolState.historyTurns,
    context: contextTurns,
    model_type,
    ...getToolPayloadState(),
    // Proactive-flow override: user opted into email up-front, so
    // force this request to take the queued path regardless of model
    // readiness. Spread after getToolPayloadState so it wins.
    ...(proactiveForceQueue ? { force_queue: 'true' } : {}),
  };

  console.log('[Chat/A-B → Flask] payload:', payload);

  // Start “busy” immediately so the user gets feedback right away
  const initialLabel =
    toolState.mode === 'search'
      ? 'Searching'
      : toolState.mode === 'ab'
        ? 'Running A/B'
        : 'Thinking';

  const endStatus = beginStatus(initialLabel);
  requestInFlight = true;
  // We don't clear activeJobs here — earlier in-flight jobs keep polling
  // independently. setActiveJobIdHint() is updated as each job starts/ends.
  setActiveJobIdHint();

  // Placeholder bot message for the chat branch. Created before apiChat so the
  // user has visible feedback in the main chat thread during the long wait,
  // then mutated in place when the response (or an error) arrives.
  let chatBotUI = null;
  const WAITING_MSG =
    'Asking Seeds of Truth AI model... This may take 1-3 minutes...';

  try {
    // Enforce gate client-side (server enforces too)
    if (!isUnlocked && (toolState.mode === 'chat' || toolState.mode === 'ab')) {
      toolState.mode = 'search';
      renderToolState();
      saveToolState();
      setNoteMessage('Searching', { busy: true });
    }

    if (toolState.mode === 'search') {
      const searchPayload = {
        user_id: userId,
        query: text,
        max_n: CFG.MAX_REFS,
        mode: 'search',
        history_turns: toolState.historyTurns,
        context: contextTurns,
        ...getToolPayloadState(),
      };

      console.log('[Search → Flask] payload:', searchPayload);
      console.log('[Chat/A-B → Flask] payload:', payload);

      // Use an inner try/finally so we *always* end busy even though we return early
      try {
        setNoteMessage('Searching corpus…', { busy: true });

        const data = await apiSearch(searchPayload);
        console.log('[Search ← Flask] response:', data);

        setNoteMessage('Rendering results…', { busy: true });

        const bot =
          data.message ||
          `Found ${data.num_results ?? data.results?.length ?? 0} items. Results below`;

        appendMessage(bot, 'bot');
        pushTurn(text, bot);

        const refs = (Array.isArray(data.references) ? data.references : [])
          .slice(0, CFG.MAX_REFS)
          .map((r) => ({
            title: r.title || r.source_title || r.filename || 'Reference',

            // Link target
            source_url: r.source_url || r.url || r.link || '',
            publisher: r.publisher || r.publication || r.source || 'Corpus',
            found_on: r.found_on || '',

            // Snippet
            snippet_html: r.snippet_html || '',
            snippet: r.snippet || r.text || r.excerpt || '',

            // Optional extras
            date: r.date || r.Date || '',
            dataset: r.dataset || '',
            subset: r.subset || '',
            score_bm25: r.score_bm25 || '',
            entity_score: r.entity_score || '',
            fulltext_score: r.fulltext_score || '',
            raw_score: r.raw_score || '',
          }));

        setReferences(refs);
      } catch (err) {
        console.error('[Search] apiSearch failed:', err);
        const serverMsg = String(err?.message || err || '').trim();
        appendMessage(
          serverMsg || 'Search request failed. Check server logs.',
          'bot',
        );
        pushStatusMessage(serverMsg || 'Search request failed.');
        setReferences([]);
      } finally {
        requestInFlight = false;
        // Search has no job_id; just refresh the hint in case other chat
        // jobs are still polling.
        setActiveJobIdHint();
        endStatus();
      }

      return;
    }

    if (toolState.mode === 'ab') {
      setNoteMessage('Generating two responses…', { busy: true });

      const data = await apiAB(payload);

      setNoteMessage('Rendering A/B…', { busy: true });

      appendABMessage(
        data.a || '',
        data.b || '',
        data.job_id_a || CFG.JOB_ID_NONE,
        data.job_id_b || CFG.JOB_ID_NONE,
        {
          labelA: data.labelA,
          labelB: data.labelB,
        },
      );

      pushTurn(
        text,
        `Response A:\n${data.a || ''}\n\nResponse B:\n${data.b || ''}`,
      );
      setReferences(Array.isArray(data.references) ? data.references : []);
      return;
    }

    // chat — async path. /api/chat returns 202 with {job_id, status:"queued"}
    // immediately. We render the placeholder bubble + waiting copy, then
    // pollJob fills it in when the worker finishes.
    setNoteMessage('Generating response…', { busy: true });

    // Create the placeholder bubble before we know the job_id so the user
    // sees in-thread feedback right away. We re-key the bubble to the real
    // job_id once /api/chat responds.
    chatBotUI = appendBotTypingMessage('', CFG.JOB_ID_NONE);
    await typeIntoElement(chatBotUI.textEl, WAITING_MSG, {
      cps: 60,
      chunkMin: 1,
      chunkMax: 4,
      maxTyped: 800,
    });

    // Streaming path: for stream-capable adapters, render tokens live (as
    // markdown) over SSE. The server falls back to the queue (a `queued`
    // event) when the model isn't ready, in which case we resume the normal
    // poll path below. Non-streaming model types skip this entirely.
    if (STREAMING_MODEL_TYPES.has(payload.model_type)) {
      const handled = await runChatStream(payload, text, chatBotUI, userId);
      if (handled) return;  // streamed (or errored) to completion
      // handled === false => server queued the job; fall through to polling
      // using the job_id it stashed on `payload.__queued`.
    }

    const data = payload.__queued || (await apiChat(payload));
    const job_id = data.job_id || CFG.JOB_ID_NONE;
    const queuedUserId = data.user_id || getUserId();

    if (!job_id || job_id === CFG.JOB_ID_NONE) {
      throw new Error('Server did not return a job_id; cannot poll for response.');
    }

    console.log(`Chat queued with job_id=${job_id} position=${data.queue_position}`);

    // Re-key the placeholder bubble to the real job_id so feedback widgets
    // and references find it later.
    if (chatBotUI?.textEl) {
      chatBotUI.textEl.setAttribute('data-job-id', job_id);
    }

    // Track this job for the global polling cadence + reload-resume.
    activeJobs.set(job_id, { placeholder: chatBotUI, startedAt: Date.now() });
    setActiveJobIdHint();
    persistActiveJob(job_id, queuedUserId);

    setEndpointStatus('starting');
    pushStatusMessageDedup(NOT_READY_MSG, 'queued');

    // Email-offer handling — preserved from the previous 503 flow.
    const offerEmail = !!data.email_offer;
    if (preSubmitEmail && job_id && queuedUserId) {
      const emailToAttach = preSubmitEmail;
      preSubmitEmail = null;  // consume once
      apiEmailResponse(job_id, queuedUserId, emailToAttach)
        .then((emailData) => {
          recordPendingEmailLS(job_id, emailToAttach);
          pendingEmails.set(job_id, { email: emailToAttach });
          const wasSentNow = emailData?.status === 'sent_now';
          setBubbleTextForJob(
            job_id,
            wasSentNow
              ? `Sent to ${emailToAttach}. (Response will also appear here if you keep the tab open.)`
              : `We'll email this response to ${emailToAttach} when it's ready. You can close the tab.`
          );
          pushStatusMessage(`Email response queued: ${emailToAttach}`);
        })
        .catch((attachErr) => {
          console.warn('[email] proactive auto-attach failed:', attachErr);
          pushStatusMessage(`Could not save email request: ${attachErr?.message || attachErr}`);
        });
    } else if (preSubmitDeclined) {
      preSubmitDeclined = false;
    } else if (offerEmail && job_id && queuedUserId) {
      openEmailModal({
        mode: 'post_queue',
        job_id,
        user_id: queuedUserId,
        queue_reason: data.queue_reason || null,
        queue_position: data.queue_position || null,
        resolveDecision: () => {},
      });
    }

    // Block until the worker finishes (or pollJob's hard timeout fires).
    // The polling itself uses /api/job/<id>; each request is fast, so we
    // never hold open a long HTTP connection that can hit Cloudflare's
    // 524 timeout.
    const result = await pollJob(
      job_id, queuedUserId, chatBotUI, data.poll_interval_ms
    );

    setNoteMessage('Rendering response…', { busy: true });

    if (result.status === 'done') {
      const finalReply = result.reply || '(no reply)';
      // Wipe the "waiting" copy and type the real reply into the same bubble.
      if (chatBotUI?.textEl) chatBotUI.textEl.textContent = '';
      await typeIntoElement(chatBotUI.textEl, finalReply, {
        cps: 60,
        chunkMin: 1,
        chunkMax: 4,
        maxTyped: 800,
      });
      chatBotUI.setSnippet?.(finalReply);
      pushTurn(text, finalReply);
      setReferences(Array.isArray(result.references) ? result.references : []);
    } else {
      // failed / timeout — surface a useful error in the bubble
      const detail = result.detail || result.error || 'Chat failed.';
      if (chatBotUI?.textEl) chatBotUI.textEl.textContent = '';
      await typeIntoElement(chatBotUI.textEl, detail, {
        cps: 80, chunkMin: 1, chunkMax: 4, maxTyped: 800,
      });
      pushStatusMessage(detail);
      setReferences([]);
    }

  } catch (err) {
    if (err?.status === 403) {
      // Access revoked / expired
      setModeAccess(false);
      toolState.mode = 'search';
      renderToolState();
      saveToolState();
      // If a chat placeholder was created before the gate revoked us,
      // mutate it in place; otherwise append a fresh bubble.
      const gateMsg = 'Not today. Search mode only.';
      if (chatBotUI?.textEl) {
        chatBotUI.textEl.textContent = gateMsg;
      } else {
        appendMessage(gateMsg, 'bot', CFG.JOB_ID_NONE);
      }
      pushTurn(text, gateMsg);
      return;
    }

    // /api/chat failed (network, 429, 500, etc.). If we already showed
    // a placeholder with the "Asking…" copy, replace its text in place
    // — appending a second bubble would leave the user staring at a
    // stale "Asking Seeds of Truth AI model…" forever.
    const serverMsg = String(err?.message || err || '').trim();
    const errorText = serverMsg || 'Error contacting server. Please try again.';
    if (chatBotUI?.textEl) {
      chatBotUI.textEl.textContent = errorText;
    } else {
      appendMessage(errorText, 'bot', CFG.JOB_ID_NONE);
    }
    pushStatusMessage(errorText);
    setReferences([]);
  } finally {
    // Covers chat + ab + outer errors.
    // Search branch returns early, but it already calls endStatus() in its own finally.
    requestInFlight = false;
    setActiveJobIdHint();
    endStatus();
  }
}

// updateMessageByJobId / processQueuedMsg were retired as part of the
// async-chat migration — bubbles are now updated directly by pollJob().
// See seedsoftruth_async_chat_design.md §7.3.

/* =========================================================
   17) ABOUT MODAL and PING TEST
   ========================================================= */
/**
 * Wires the About modal's open/close buttons, overlay-click dismissal,
 * and Escape-key handling.
 * @returns {void}
 */
function initAboutModal() {
  const btn = document.getElementById('about-btn');
  const overlay = document.getElementById('about-overlay');
  const closeBtn = document.getElementById('about-close');

  if (!btn || !overlay || !closeBtn) return;

  function open() {
    overlay.classList.add('show');
    overlay.setAttribute('aria-hidden', 'false');
  }

  function close() {
    overlay.classList.remove('show');
    overlay.setAttribute('aria-hidden', 'true');
  }

  btn.addEventListener('click', open);
  closeBtn.addEventListener('click', close);

  // click outside
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) close();
  });

  // escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && overlay.classList.contains('show')) close();
  });
}

/**
 * Wires the ping-test button to POST /api/ping and display the result.
 * @returns {void}
 */
function initPingTest() {
  const btn = document.getElementById('ping-btn');
  const output = document.getElementById('ping-result');
  if (!btn || !output) return;

  btn.addEventListener('click', async () => {
    output.textContent = 'Sending request...';

    try {
      const data = await apiPost(CFG.API.PING, {
        from: 'browser',
        test: 'JS → Flask → JS',
      });
      output.textContent = JSON.stringify(data, null, 2);
    } catch (err) {
      output.textContent = 'Error: ' + (err?.message || String(err));
    }
  });
}

/* =========================================================
   18) RANGE FILL + AUTOSIZE (helpers)
   ========================================================= */
/**
 * Wires the history-slider range input so its filled-track CSS variable
 * tracks the current value.
 * @returns {void}
 */
function initRangeFill() {
  const r = document.getElementById('history-slider');
  if (!r) return;

  function setFill() {
    const min = Number(r.min || 0);
    const max = Number(r.max || 100);
    const val = Number(r.value || 0);
    const pct = ((val - min) / (max - min)) * 100;
    r.style.setProperty('--fill', pct + '%');
  }

  r.addEventListener('input', setFill);
  setFill();
}

/**
 * Wires the chat input textarea to auto-resize on input and window resize.
 * @returns {void}
 */
function initAutosizeTextarea() {
  const ta = document.getElementById('chat-input');
  if (!ta) return;
  ta.addEventListener('input', autoResizeTextarea);
  window.addEventListener('resize', autoResizeTextarea);
  autoResizeTextarea();
}

/* =========================================================
   19) LOCK UI
   ========================================================= */

/**
 * Closes the lock (password) modal and clears its message/password field.
 * @returns {void}
 */
function closeLockModal() {
  const lockModal = document.getElementById('lock-modal');
  const lockPass = document.getElementById('lock-pass');
  const lockMsg = document.getElementById('lock-msg');

  if (!lockModal) return;

  lockModal.classList.remove('open');
  lockModal.setAttribute('aria-hidden', 'true');

  if (lockMsg) lockMsg.textContent = '';
  if (lockPass) lockPass.value = '';
}

/**
 * Sets the global unlocked state and updates the mode radios, lock
 * button, and tool state to match (forcing search mode when locked).
 * @param {boolean} unlocked Whether chat/A-B access is granted.
 * @returns {void}
 */
function setModeAccess(unlocked) {
  isUnlocked = !!unlocked;

  const modeSearch = document.getElementById('mode-search');
  const modeChat = document.getElementById('mode-chat');
  const modeAb = document.getElementById('mode-ab');
  const lockBtnEl = document.getElementById('lock-btn');

  if (modeChat) modeChat.disabled = !isUnlocked;
  if (modeAb) modeAb.disabled = !isUnlocked;

  if (!isUnlocked) toolState.mode = 'search';
  if (!isUnlocked && modeSearch) modeSearch.checked = true;

  if (lockBtnEl) {
    lockBtnEl.classList.toggle('unlocked', isUnlocked);
    lockBtnEl.disabled = isUnlocked;
    lockBtnEl.setAttribute(
      'aria-label',
      isUnlocked ? 'Access granted. Connecting...' : 'Restricted access',
    );
  }

  renderToolState();
  saveToolState();

  // close after any re-render, using global re-querying closer
  if (isUnlocked) closeLockModal();
}

/**
 * Wires the lock (password) UI: opening the modal, submitting the
 * password to /api/unlock, and handling success (granting access and,
 * when the model is ready, switching to chat mode) and failure.
 * @returns {void}
 */
function initLockUI() {
  const lockBtn = document.getElementById('lock-btn');
  const lockModal = document.getElementById('lock-modal');
  const lockClose = document.getElementById('lock-close');
  const lockEnter = document.getElementById('lock-enter');
  const lockPass = document.getElementById('lock-pass');
  const lockMsg = document.getElementById('lock-msg');

  if (
    !lockBtn ||
    !lockModal ||
    !lockClose ||
    !lockEnter ||
    !lockPass ||
    !lockMsg
  )
    return;

  /**
   * Opens the lock modal, clears its fields, and focuses the password input.
   * @returns {void}
   */
  function openLockModal() {
    lockModal.classList.add('open');
    lockModal.setAttribute('aria-hidden', 'false');
    lockMsg.textContent = '';
    lockPass.value = '';
    setTimeout(() => lockPass.focus(), 0);
  }

  /**
   * Reads the entered password, submits it to /api/unlock, and grants or
   * denies access based on the response.
   * @returns {Promise<void>} Resolves when the unlock attempt completes.
   */
  async function tryUnlock() {
    const pw = (lockPass.value || '').trim();
    if (!pw) return;

    lockEnter.disabled = true;
    lockPass.disabled = true;
    lockMsg.textContent = 'Checking…';

    try {
      const data = await apiUnlock(pw);
      const unlocked = !!(data && (data.unlocked === true || data.ok === true));

      if (unlocked) {
        setModeAccess(true);
        // On a successful unlock, default the user into AI chat mode —
        // but only when the model endpoint is known-ready. If it's still
        // warming (or readiness is unknown), leave them in search mode so
        // they aren't dropped into a chat the backend can't answer yet.
        if (lastKnownSparkReady === true) {
          toolState.mode = 'chat';
          renderToolState();
          saveToolState();
        }
      } else {
        setModeAccess(false);
        lockMsg.textContent =
          (data && (data.message || data.error)) || 'Incorrect password';
      }
    } catch (err) {
      setModeAccess(false);
      lockMsg.textContent = err?.message || 'Not today';
    } finally {
      lockEnter.disabled = false;
      lockPass.disabled = false;
    }
  }

  lockBtn.addEventListener('click', openLockModal);
  lockClose.addEventListener('click', closeLockModal);
  lockModal.addEventListener('click', (e) => {
    if (e.target === lockModal) closeLockModal();
  });
  lockEnter.addEventListener('click', tryUnlock);

  lockPass.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault(); // prevents newline / implicit form submit
      tryUnlock();
    }
    if (e.key === 'Escape') closeLockModal();
  });
}

/* =========================================================
   20) STATUS + QUEUE POLLING
   ========================================================= */

/**
 * Pushes a status message only when it differs from the last message
 * shown under the same key, to avoid spamming the panel on every poll.
 * @param {string} msg The status message text.
 * @param {string=} key Dedup key grouping related messages.
 * @returns {void}
 */
function pushStatusMessageDedup(msg, key = 'generic') {
  const map =
    pushStatusMessageDedup._map || (pushStatusMessageDedup._map = new Map());
  if (map.get(key) === msg) return;
  map.set(key, msg);
  pushStatusMessage(msg);
}

let lastHealthCheckAt = 0;

/**
 * Polls /api/status once and updates the endpoint indicator, model
 * readiness cache, and access state. Optionally requests a health check;
 * avoids clobbering UI state while a request is in flight.
 * @param {boolean=} sendHealth Force a health check on this poll.
 * @returns {Promise<void>} Resolves when the poll completes.
 */
async function refreshStatusOnce(sendHealth = false) {
  try {
    const now = Date.now();
    const doHealth = sendHealth || now - lastHealthCheckAt > 60_000;

    const data = await apiStatus({
      health: doHealth,
      model_type: toolState.modelType,
    });

    if (typeof data?.unlocked === 'boolean') setModeAccess(data.unlocked);

    const indexReady = data?.retrieval_state_ready === true;
    const hasModelReady = 'model_ready' in data;
    const modelReady = data?.model_ready === true;

    // Only feed the Spark-readiness cache when this poll actually targeted
    // Spark. Otherwise we'd store some other backend's readiness under a
    // Spark-named variable and corrupt the fallback routing decision.
    if (hasModelReady && toolState.modelType === PRIMARY_MODEL_TYPE) {
      lastKnownSparkReady = modelReady;
      lastSparkHealthCheckAt = now;
    }

    if (doHealth && hasModelReady) lastHealthCheckAt = now;

    // During an active request, do NOT let polling replace the main UI state.
    // Only update the endpoint chip if the endpoint is actually healthy.
    if (requestInFlight) {
      if (indexReady && hasModelReady && modelReady) {
        setEndpointStatus('ready');
        els.endpointLabel.textContent = 'Endpoint Ready';
        els.endpointChip.textContent = 'healthy';
      }
      return;
    }

    if (doHealth && typeof data?.model_ready === 'boolean') {
      pushStatusMessageDedup(
        `Model health: ${data.model_ready ? 'OK' : 'not ready'}`,
        'model',
      );
    }

    if (indexReady && hasModelReady && modelReady) {
      setEndpointStatus('ready');
      els.endpointLabel.textContent = 'Endpoint Ready';
      els.endpointChip.textContent = 'healthy';
    } else if (indexReady && hasModelReady && !modelReady) {
      setEndpointStatus('starting');
      els.endpointLabel.textContent = 'Index ready, model warming…';
      els.endpointChip.textContent = 'warming';
    } else if (hasModelReady) {
      setEndpointStatus('starting');
      els.endpointLabel.textContent = 'Endpoint starting…';
      els.endpointChip.textContent = 'starting';
    }
  } catch (err) {
    console.error('problem in refreshStatusOnce: ', err.message);

    // Don't flip to "off" while a real request is in flight.
    if (!requestInFlight) {
      setEndpointStatus('off');
    }
  }
}

/**
 * Polls /api/queue once: updates the queue count display, caches queue
 * depth and email-offer state for the proactive email flow, and applies
 * any delayed queued chat response carried in the response.
 * @returns {Promise<void>} Resolves when the poll completes.
 */
async function refreshQueueOnce() {
  try {
    const data = await apiQueue({ health: false }); // let status line handle health

    const q = Number(data?.queries_in_line ?? 0);
    setQueueStatus(q);

    // Cache for the proactive pre-submit email check. The server is
    // the source of truth on email_offer — we never assume true.
    lastKnownQueueDepth = q;
    lastKnownEmailOffer = !!data?.email_offer;

    // /api/queue is now a pure depth widget; per-job answers come via
    // /api/job/<id> polled by pollJob(). The legacy outgoing_resp branch
    // has been removed — see seedsoftruth_async_chat_design.md §6.4.
    if (typeof data?.queries_in_line === 'number') {
      pushStatusMessageDedup(
        `Queue: ${q} waiting`,
        'queue',
      );
    }
  } catch (e) {
    console.error('refreshQueueOnce error:', e.message);

    if (!requestInFlight) {
      setEndpointStatus('off');
      if (els.queueCountEl) els.queueCountEl.textContent = '—';
      if (els.queueEtaEl) els.queueEtaEl.textContent = '—';
      pushStatusMessageDedup(
        'Queue check failed (endpoint unreachable).',
        'queue-error',
      );
    }
  }
}

/**
 * Runs a single status and queue refresh.
 * @param {boolean=} sendHealth Force a health check on the status poll.
 * @returns {void}
 */
function checkStatusAndQueue(sendHealth = true) {
  refreshStatusOnce(sendHealth);
  refreshQueueOnce();
}

/**
 * Starts the continual background polling loop for endpoint status and
 * the queue (the latter also picking up delayed responses). Polls more
 * thoroughly while a request or job is active.
 * @returns {void}
 */
function startPolling() {
  setInterval(
    () => {
      if (requestInFlight || activeJobId) {
        refreshQueueOnce();
        refreshStatusOnce();
      } else {
        refreshStatusOnce(false);
      }
    },
    Math.min(CFG.QUEUE_POLL_MS, CFG.STATUS_POLL_MS),
  );
}

/* =========================================================
   21) INIT / WIRING (bottom)
   ========================================================= */

/**
 * Caches references to all DOM elements the app uses into the `els`
 * object. Must run before any code that reads from `els`.
 * @returns {void}
 */
function initDom() {
  els.body = document.body;

  // core chat
  els.chatForm = document.getElementById('chat-form');
  els.chatInput = document.getElementById('chat-input');
  els.messagesEl = document.getElementById('messages');
  els.chatContainer = document.getElementById('chat-container');
  els.welcomeMessage = document.getElementById('welcome-message');

  // theme + sidebar
  els.themeToggleButtons = Array.from(
    document.querySelectorAll('[data-theme-toggle]'),
  );
  els.sidebar = document.getElementById('sidebar');
  els.menuBtn = document.getElementById('menu-btn');
  els.overlay = document.getElementById('overlay');
  els.sidebarOpenBtn = document.getElementById('sidebar-open-btn');
  els.sidebarCollapseBtn = document.getElementById('sidebar-collapse-btn');

  // tools
  els.toolsBtn = document.getElementById('tools-btn');
  els.toolsPopup = document.getElementById('tools-popup');

  // refs
  els.referencesContainer = document.getElementById('references-container');
  els.referencesCount = document.getElementById('references-count');
  els.referencesEmpty = document.getElementById('references-empty');
  els.referencesTitle = document.getElementById('references-title');

  // tool controls
  els.historySlider = document.getElementById('history-slider');
  els.historyValue = document.getElementById('history-value');
  els.historyHelpN = document.getElementById('history-help-n');
  els.modeHelp = document.getElementById('mode-help');
  els.modeRadios = Array.from(document.querySelectorAll('input[name="mode"]'));
  els.searchGroup = document.getElementById('search-group');
  els.ragToggle = document.getElementById('rag-toggle');
  els.ragAlgoType = document.getElementById('rag-algo-type');
  els.promptType = document.getElementById('prompt-type');
  els.forceQueueToggle = document.getElementById('force-queue-toggle');

  // status panel
  els.endpointDot = document.getElementById('endpoint-dot');
  els.endpointLabel = document.getElementById('endpoint-label');
  els.endpointChip = document.getElementById('endpoint-chip');
  els.queueCountEl = document.getElementById('queue-count');
  els.queueEtaEl = document.getElementById('queue-eta');
  els.statusMessagesEl = document.getElementById('status-messages');

  // saved convos
  els.saveConvoBtn = document.getElementById('save-convo-btn');
  els.recentList = document.getElementById('recent-list');

  // trash chat
  els.trashChatBtn = document.getElementById('trash-chat-btn');

  // about
  els.aboutBtn = document.getElementById('about-btn');

  // feedback modal
  els.fbOverlay = document.getElementById('fb-overlay');
  els.fbClose = document.getElementById('fb-close');
  els.fbCancel = document.getElementById('fb-cancel');
  els.fbSubmit = document.getElementById('fb-submit');
  els.fbMeta = document.getElementById('fb-meta');
  els.fbAccuracy = document.getElementById('fb-accuracy');
  els.fbStyle = document.getElementById('fb-style');
  els.fbRelevance = document.getElementById('fb-relevance');
  els.fbComments = document.getElementById('fb-comments');
  els.fbToast = document.getElementById('fb-toast');
  els.fbFieldAccuracy = document.getElementById('fb-field-accuracy');
  els.fbFieldStyle = document.getElementById('fb-field-style');
  els.fbJobId = document.getElementById('fb-job-id');

  // custom modal
  els.modalOverlay = document.getElementById('modal-overlay');
  els.modalTitle = document.getElementById('modal-title');
  els.modalMessage = document.getElementById('modal-message');
  els.modalActions = document.getElementById('modal-actions');
  els.modalCloseBtn = document.getElementById('modal-close');

  // input note status
  els.noteTextWrap = document.getElementById('note-text');
  els.noteMessage = document.getElementById('note-message');
  els.noteSpinner = document.getElementById('note-spinner');

  // version info
  els.versionNum = document.getElementById('sidebar-version-num');
  els.versionName = document.getElementById('sidebar-version-name');
}

/**
 * Writes the configured version number and name into the sidebar.
 * @returns {void}
 */
function initVersion() {
  els.versionNum.textContent = CFG.VERSION_NUM;
  els.versionName.textContent = CFG.VERSION_NAME;
}

/**
 * Wires up the remaining top-level event handlers: modal close, chat form
 * submit and Enter-to-send, initial input focus, and the save/trash chat
 * buttons.
 * @returns {void}
 */
function initWiring() {
  // modal overlay close
  if (els.modalOverlay) {
    els.modalOverlay.addEventListener('click', (e) => {
      if (e.target === els.modalOverlay) closeModal(false);
    });
  }
  if (els.modalCloseBtn) {
    els.modalCloseBtn.addEventListener('click', () => closeModal(false));
  }

  // chat
  if (els.chatForm) els.chatForm.addEventListener('submit', handleChatSubmit);
  if (els.chatInput) {
    els.chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        els.chatForm.requestSubmit();
      }
    });
  }

  // focus cursor on load
  if (els.chatInput) {
    els.chatInput.focus();
    try {
      els.chatInput.setSelectionRange(
        els.chatInput.value.length,
        els.chatInput.value.length,
      );
    } catch (_) {}
  }

  // save convo
  if (els.saveConvoBtn)
    els.saveConvoBtn.addEventListener('click', saveCurrentConversation);

  // trash chat
  if (els.trashChatBtn)
    els.trashChatBtn.addEventListener('click', clearCurrentChat);
}

/* =========================================================
   20) EMAIL-RESPONSE FLOW
   When a chat request gets queued (HTTP 503) AND the server's
   queued response includes email_offer:true, we open a modal
   asking the user whether they want the response emailed.
   ========================================================= */

// In-memory registry: job_id -> { email, bubbleEl, statusEl }.
// Lets us flip pending-email bubbles into "delivered" state if the
// real response also arrives in-tab (the user explicitly opted into
// both: visible in-tab AND emailed).
const pendingEmails = new Map();

// Snapshot of the most recent /api/queue poll. Used to decide whether
// the proactive pre-submit email modal should fire. Updated in
// refreshQueueOnce. These are intentionally module-level so the
// pre-submit check in handleChatSubmit is cheap (no API call).
let lastKnownQueueDepth = 0;
let lastKnownEmailOffer = false;

// Stash for the proactive (pre-submit) flow. Set by handleChatSubmit
// after the user picks "Email me" up-front; consumed by the 503 catch
// to auto-attach without re-prompting. preSubmitDeclined suppresses
// the post-queue modal when the user explicitly chose to wait.
let preSubmitEmail = null;
let preSubmitDeclined = false;

/**
 * Loads the pending-email list from localStorage, dropping entries older
 * than PENDING_EMAIL_TTL_MS on read.
 * @returns {!Array<{job_id: string, email: string, submitted_at: number}>}
 *     The non-expired pending-email records.
 */
function loadPendingEmailsLS() {
  try {
    const raw = localStorage.getItem(CFG.LS_PENDING_EMAILS);
    if (!raw) return [];
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) return [];
    const now = Date.now();
    // Drop expired entries on read; keeps the list bounded without a
    // separate sweep. TTL covers the case where the server completed
    // and emailed already but we never got a tab signal.
    return arr.filter(
      (e) =>
        e && e.submitted_at && now - e.submitted_at < CFG.PENDING_EMAIL_TTL_MS,
    );
  } catch (_) {
    return [];
  }
}

/**
 * Persists the pending-email list to localStorage.
 * @param {!Array<Object>} arr The pending-email records to store.
 * @returns {void}
 */
function savePendingEmailsLS(arr) {
  try {
    localStorage.setItem(CFG.LS_PENDING_EMAILS, JSON.stringify(arr));
  } catch (_) {}
}

/**
 * Records a pending email for a job id in localStorage, replacing any
 * existing entry for the same job id.
 * @param {string} job_id The queued job's id.
 * @param {string} email The email address awaiting the response.
 * @returns {void}
 */
function recordPendingEmailLS(job_id, email) {
  const list = loadPendingEmailsLS().filter((e) => e.job_id !== job_id);
  list.push({ job_id, email, submitted_at: Date.now() });
  savePendingEmailsLS(list);
}

/**
 * Removes the pending-email entry for a job id from localStorage.
 * @param {string} job_id The queued job's id.
 * @returns {void}
 */
function removePendingEmailLS(job_id) {
  const list = loadPendingEmailsLS().filter((e) => e.job_id !== job_id);
  savePendingEmailsLS(list);
}

/**
 * Persists the last-used email address for autofill convenience.
 * @param {string} email The email address to remember.
 * @returns {void}
 */
function rememberLastEmail(email) {
  try {
    localStorage.setItem(CFG.LS_LAST_EMAIL, email);
  } catch (_) {}
}

/**
 * Reads the last-used email address from localStorage.
 * @returns {string} The remembered email, or '' when none is stored.
 */
function recallLastEmail() {
  try {
    return localStorage.getItem(CFG.LS_LAST_EMAIL) || '';
  } catch (_) {
    return '';
  }
}

// Email modal state. The "current" object is whatever the active
// queueing event is asking the user about — only one at a time.
// Shape: {
//   mode: 'post_queue' | 'pre_submit',
//   job_id?,                  // present in post_queue mode
//   user_id?,                 // present in post_queue mode
//   queue_reason?, queue_position?,
//   resolveDecision(decision) // called with { decision, email? }
// }
//
// post_queue (existing): we already have a job_id from a 503; submit
//   calls /api/email_response immediately.
// pre_submit (new): we don't have a job_id yet. submit captures the
//   email and resolves the promise so the caller can include
//   force_queue=true in the chat request, then auto-attach the email
//   when the 503 comes back.
let activeEmailRequest = null;

// Helper: update the bot bubble for a given job_id. The bubble is
// keyed by data-job-id (set in appendMessage), so any code path that
// has the job_id can flip the text without holding an element ref.
/**
 * Updates the bot chat bubble keyed by data-job-id, if it exists.
 * @param {string} job_id The job id identifying the bubble.
 * @param {string} text The new bubble text.
 * @returns {boolean} True if a bubble was found and updated.
 */
function setBubbleTextForJob(job_id, text) {
  if (!job_id) return false;
  const el = document.querySelector(`.message-text[data-job-id="${job_id}"]`);
  if (!el) return false;
  el.textContent = text;
  return true;
}

// Pick the title + subtitle the modal should show given the server's
// reason for queueing. Two scenarios distinguished today:
//   - model_warming: the model is cold / not yet online. User wait
//                    depends on warm-up timing, not on a backlog.
//   - queue_busy:    model is online but there's a backlog of requests
//                    ahead. Wait scales with queue depth + inference time.
// queue_position is the 1-indexed slot of this request in the queue;
// "ahead = position - 1" since the user occupies position itself.
/**
 * Chooses the email-modal title and subtitle based on why the request
 * was queued (model warming vs. a busy queue) and the queue position.
 * @param {?string} queue_reason The server's queue reason
 *     ('model_warming', 'queue_busy', or unknown).
 * @param {?number} queue_position The 1-indexed slot of this request.
 * @returns {{title: string, subtitle: string}} The modal copy.
 */
function emailModalCopyForReason(queue_reason, queue_position) {
  const pos =
    typeof queue_position === 'number' && queue_position > 0
      ? queue_position
      : null;
  const ahead = pos ? Math.max(0, pos - 1) : null;

  if (queue_reason === 'model_warming') {
    return {
      title: 'Waiting for the model',
      subtitle:
        "The model isn't online yet — it needs a minute or two to warm up. " +
        "We can email you the response when it's ready, or you can wait here.",
    };
  }

  // 'queue_busy' (or unknown reason — fall through here).
  if (ahead && ahead > 0) {
    const noun = ahead === 1 ? 'request' : 'requests';
    return {
      title: 'This will take a few minutes',
      subtitle:
        `There ${ahead === 1 ? 'is' : 'are'} ${ahead} ${noun} ahead of yours, ` +
        'and inference can take 2–5 minutes each. We can email you the ' +
        "response when it's ready, or you can wait here.",
    };
  }

  // Default queue_busy phrasing for position 1 / unknown.
  return {
    title: 'This will take a few minutes',
    subtitle:
      "Inference can take 2–5 minutes. We can email you the response when it's " +
      'ready, or you can wait here.',
  };
}

/**
 * Opens the email-response modal for a queueing event, setting the
 * reason-specific copy and prefilling the last-used email. If the modal
 * elements are missing, resolves the request's decision as 'wait'.
 * @param {{mode: string, job_id?: string, user_id?: string,
 *     queue_reason?: ?string, queue_position?: ?number,
 *     resolveDecision: function(*): void}} req The active email request.
 * @returns {void}
 */
function openEmailModal(req) {
  const modal = document.getElementById('email-modal');
  const input = document.getElementById('email-input');
  const submit = document.getElementById('email-submit');
  const wait = document.getElementById('email-wait');
  const close = document.getElementById('email-close');
  const msg = document.getElementById('email-msg');
  const titleEl = document.getElementById('email-title');
  const subEl = document.getElementById('email-subtitle');
  if (!modal || !input || !submit || !wait || !close || !msg) {
    console.warn('[email] modal elements missing — skipping email offer');
    if (req && typeof req.resolveDecision === 'function')
      req.resolveDecision('wait');
    return;
  }

  activeEmailRequest = req;

  // Pick title/subtitle from queue_reason — keeps the copy honest about
  // what the user is actually waiting on.
  const copy = emailModalCopyForReason(req?.queue_reason, req?.queue_position);
  if (titleEl) titleEl.textContent = copy.title;
  if (subEl) subEl.textContent = copy.subtitle;

  input.value = recallLastEmail();
  msg.textContent = '';
  submit.disabled = false;
  input.disabled = false;

  modal.classList.add('open');
  modal.setAttribute('aria-hidden', 'false');
  setTimeout(() => input.focus(), 0);
}

/**
 * Hides the email-response modal.
 * @returns {void}
 */
function closeEmailModal() {
  const modal = document.getElementById('email-modal');
  if (!modal) return;
  modal.classList.remove('open');
  modal.setAttribute('aria-hidden', 'true');
}

/**
 * Performs a lightweight client-side check that a string looks like a
 * valid email address (mirrors the server-side regex; UX gate only).
 * @param {*} s Candidate email string.
 * @returns {boolean} True if the string looks like a valid email.
 */
function _isLikelyValidEmail(s) {
  if (!s || typeof s !== 'string') return false;
  const cleaned = s.trim();
  // Mirror the server-side regex closely; this is just a UX gate.
  return (
    /^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$/.test(cleaned) &&
    cleaned.length <= 254
  );
}

/**
 * Handles the email-modal submit. Validates the address, then either
 * (pre_submit mode) captures the email and resolves the caller's promise
 * so the chat can be force-queued, or (post_queue mode) immediately calls
 * /api/email_response to attach the email to the known job id.
 * @returns {Promise<void>} Resolves when the submit flow completes.
 */
async function submitEmailRequest() {
  const req = activeEmailRequest;
  if (!req) return;

  const input = document.getElementById('email-input');
  const submit = document.getElementById('email-submit');
  const msg = document.getElementById('email-msg');

  const email = (input?.value || '').trim();
  if (!_isLikelyValidEmail(email)) {
    msg.textContent = "That doesn't look like a valid email.";
    return;
  }

  // pre_submit: we don't have a job_id yet. Just capture the email and
  // hand control back to the caller (handleChatSubmit) so it can send
  // the chat with force_queue=true. The 503 catch then auto-attaches.
  if (req.mode === 'pre_submit') {
    rememberLastEmail(email);
    closeEmailModal();
    activeEmailRequest = null;
    pushStatusMessage(
      'Email request captured — sending your question to the queue.',
    );
    if (typeof req.resolveDecision === 'function') {
      req.resolveDecision({ decision: 'emailed', email });
    }
    return;
  }

  // post_queue: job_id is known; call /api/email_response now.
  submit.disabled = true;
  input.disabled = true;
  msg.textContent = 'Submitting…';

  try {
    const data = await apiEmailResponse(req.job_id, req.user_id, email);
    rememberLastEmail(email);
    recordPendingEmailLS(req.job_id, email);

    pendingEmails.set(req.job_id, { email });

    // Reframe the in-thread bubble so the user sees a clear, stable
    // "we'll email you" message instead of the queued/waiting copy.
    const wasSentNow = data?.status === 'sent_now';
    setBubbleTextForJob(
      req.job_id,
      wasSentNow
        ? `Sent to ${email}. (Response will also appear here if you keep the tab open.)`
        : `We'll email this response to ${email} when it's ready. You can close the tab.`,
    );

    closeEmailModal();
    pushStatusMessage(`Email response queued: ${email}`);
    if (typeof req.resolveDecision === 'function') {
      req.resolveDecision({ decision: 'emailed', email });
    }
  } catch (err) {
    msg.textContent =
      err?.message || 'Could not save email request. Try again.';
    submit.disabled = false;
    input.disabled = false;
  }
}

/**
 * Dismisses the email modal without sending an email, resolving the
 * active request's decision with the given reason.
 * @param {string=} reason Decision reason (defaults to 'wait').
 * @returns {void}
 */
function dismissEmailRequest(reason) {
  const req = activeEmailRequest;
  closeEmailModal();
  activeEmailRequest = null;
  if (req && typeof req.resolveDecision === 'function') {
    req.resolveDecision({ decision: reason || 'wait' });
  }
}

// Cancel a previously-attached email request. Wired up if/when the UI
// grows a "cancel email" button per pending bubble. For now, exported
// on window for ad-hoc invocation.
/**
 * Cancels a previously-attached email request server-side and removes its
 * local pending-email state.
 * @param {string} job_id The queued job's id.
 * @param {string} user_id The requesting user's id.
 * @returns {Promise<void>} Resolves when the cancellation completes.
 */
async function cancelPendingEmail(job_id, user_id) {
  try {
    await apiEmailResponseCancel(job_id, user_id);
  } catch (err) {
    console.warn('[email] cancel failed:', err?.message || err);
  }
  pendingEmails.delete(job_id);
  removePendingEmailLS(job_id);
}

// Promise-wrapped opener for the proactive (pre_submit) flow. Resolves
// with { decision: 'emailed', email } if the user opts in, or
// { decision: 'wait' } otherwise. Exists so handleChatSubmit can await
// the decision before assembling the chat payload.
/**
 * Opens the email modal in pre-submit mode and returns a promise for the
 * user's decision, so handleChatSubmit can await it before building the
 * chat payload.
 * @param {{queue_reason?: string, queue_position?: ?number}=} opts
 *     Context for the modal copy.
 * @returns {Promise<{decision: string, email?: string}>} Resolves with
 *     {decision: 'emailed', email} on opt-in, else {decision: 'wait'}.
 */
function offerEmailPreSubmit(opts) {
  return new Promise((resolve) => {
    openEmailModal({
      mode: 'pre_submit',
      queue_reason: opts?.queue_reason || 'queue_busy',
      queue_position: opts?.queue_position || null,
      resolveDecision: (result) => resolve(result || { decision: 'wait' }),
    });
  });
}

/**
 * Wires the email-response modal: submit, wait, close, overlay-click
 * dismissal, and Enter/Escape key handling.
 * @returns {void}
 */
function initEmailModal() {
  const modal = document.getElementById('email-modal');
  const input = document.getElementById('email-input');
  const submit = document.getElementById('email-submit');
  const wait = document.getElementById('email-wait');
  const close = document.getElementById('email-close');
  if (!modal || !input || !submit || !wait || !close) return;

  submit.addEventListener('click', submitEmailRequest);
  wait.addEventListener('click', () => dismissEmailRequest('wait'));
  close.addEventListener('click', () => dismissEmailRequest('wait'));

  // Click outside the card = "I'll wait"
  modal.addEventListener('click', (e) => {
    if (e.target === modal) dismissEmailRequest('wait');
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      submitEmailRequest();
    }
    if (e.key === 'Escape') dismissEmailRequest('wait');
  });
}

// On page load, surface any pending emails from a previous session as
// a status-panel hint. We don't try to reconstruct chat bubbles here —
// the conversation thread isn't persisted across refreshes.
/**
 * On page load, surfaces any pending emails from a previous session as
 * status-panel hints. Chat bubbles are not reconstructed.
 * @returns {void}
 */
function reconcilePendingEmailsOnLoad() {
  const list = loadPendingEmailsLS();
  if (!list.length) return;
  for (const e of list) {
    pushStatusMessage(`Pending email response will be sent to ${e.email}`);
  }
}

/**
 * Application entry point: caches DOM elements, loads persisted state,
 * initializes all UI subsystems (theme, sidebar, tools, modals, lock,
 * email), checks server access, starts polling, and exposes debug hooks.
 * Runs on DOMContentLoaded.
 * @returns {void}
 */
function init() {
  initDom();

  // sanity
  if (!els.chatForm || !els.chatInput || !els.messagesEl) {
    console.warn('Seeds of Truth app.js: required chat elements not found.');
    return;
  }

  // set version info
  initVersion();

  // tool state
  loadToolState();

  // default locked until server says otherwise
  setModeAccess(false);

  // theme + sidebar + tools
  initTheme();
  initSidebarCollapse();
  initMobileSidebar();
  initToolsPopup();

  // feedback + about + wiring
  initFeedbackModal();
  initAboutModal();
  initWiring();

  // helpers
  initRangeFill();
  initAutosizeTextarea();
  initPingTest();
  renderRecentList();
  renderToolState();

  // lock modal
  initLockUI();

  // email-response modal (offered when chat 503-queues with email_offer:true)
  initEmailModal();
  reconcilePendingEmailsOnLoad();

  // ask server whether this session is already unlocked
  apiAccess()
    .then((d) => {
      const unlocked = !!d.unlocked;
      setModeAccess(unlocked);
      if (unlocked) {
        toolState.mode = 'chat';
      }
    })
    .catch(() => setModeAccess(false));

  if (!CFG.DEV_MODE) {
    document.querySelectorAll('[data-dev-only]').forEach((el) => {
      el.hidden = true;
    });
  }

  // set initial status / queue check on page load
  checkStatusAndQueue(true);

  // starts continual polling for status and delayed responses
  // @TODO: Implement something smarter. Perhaps polling need only occur if we know we have delayed responses.
  startPolling();

  // If a chat was in flight when the tab last unloaded, recreate a
  // placeholder bubble and resume polling for it. Fire-and-forget — we
  // don't want to block init on a slow network. See pollJob /
  // persistActiveJob for the persistence contract.
  try {
    resumeActiveJobIfAny();
  } catch (e) {
    console.warn('[init] resumeActiveJobIfAny failed:', e);
  }

  // expose debug hooks
  window.setReferences = setReferences;
  window.openFeedbackModal = openFeedbackModal;
  window.pushStatusMessage = pushStatusMessage;

  // enable sim + vllm adapters only for devmode. Both are appended via
  // JS rather than baked into the HTML so a non-DEV_MODE deploy can't
  // expose them via a stale option in the rendered template.
  if (CFG.DEV_MODE) {
    const select = document.getElementById('model-type');

    const simOption = document.createElement('option');
    simOption.textContent = 'Sim Adapter (testing)';
    simOption.value = 'sim';
    select.appendChild(simOption);

    // vLLM streaming adapter — talks to a vLLM /v1/chat/completions
    // endpoint with stream=true. See VLLMStreamingLLM in model_adapters.py
    // for the SSE consumption logic and Cloudflare-timeout rationale.
    const vllmOption = document.createElement('option');
    vllmOption.textContent = 'vLLM (streaming)';
    vllmOption.value = 'vllm';
    select.appendChild(vllmOption);

    // DeepInfra streaming adapter — talks to DeepInfra's OpenAI-compatible
    // /v1/openai/chat/completions endpoint with stream=true. See
    // DeepInfraStreamingLLM in model_adapters.py for the SSE consumption
    // logic. Unlike vLLM, DeepInfra is a public managed API (no Cloudflare
    // wrapper), so it sidesteps the SSE-buffering issues vLLM hit.
    const deepInfraStreamOption = document.createElement('option');
    deepInfraStreamOption.textContent = 'DeepInfra (streaming)';
    deepInfraStreamOption.value = 'deepinfra_stream';
    select.appendChild(deepInfraStreamOption);

    // Re-apply any persisted dev selection now that the dev-only options
    // exist — the earlier reconciliation (init) ran before these were
    // appended, so a cached 'vllm'/'sim'/'deepinfra_stream' wouldn't have
    // visibly selected in the dropdown.
    if (toolState.modelType) {
      select.value = toolState.modelType;
    }
  }
}

document.addEventListener('DOMContentLoaded', init);

(function clickDistortion() {
  const host = document.getElementById('click-distort');
  const svg = document.querySelector('filter#sot-displace');
  if (!host || !svg) return;

  // Find the displacement map inside the filter
  const disp = document.querySelector('#sot-displace feDisplacementMap');
  const turb = document.querySelector('#sot-displace feTurbulence');
  if (!disp || !turb) return;

  let seed = 2;

  function spawn(x, y) {
    // Lens
    const lens = document.createElement('div');
    lens.className = 'click-lens';
    lens.style.left = x + 'px';
    lens.style.top = y + 'px';

    // Ring
    const ring = document.createElement('div');
    ring.className = 'click-ring';
    ring.style.left = x + 'px';
    ring.style.top = y + 'px';

    host.appendChild(lens);
    host.appendChild(ring);

    // Vary noise a bit each click
    seed = (seed + 1) % 9999;
    turb.setAttribute('seed', String(seed));

    // Animate: bump distortion up then down quickly
    // Note: filter is shared, but we only show one lens briefly.
    disp.setAttribute('scale', '0');

    // turn on transitions next frame
    requestAnimationFrame(() => {
      lens.classList.add('on');
      ring.classList.add('on');

      // distortion punch
      disp.setAttribute('scale', '26');

      // ease back
      setTimeout(() => disp.setAttribute('scale', '0'), 140);

      // cleanup
      setTimeout(() => {
        lens.remove();
        ring.remove();
      }, 520);
    });
  }

  // Use pointerdown so it works on touch too
  window.addEventListener(
    'pointerdown',
    (e) => {
      // ignore right-click
      if (e.button === 2) return;
      spawn(e.clientX, e.clientY);
    },
    { passive: true },
  );
})();

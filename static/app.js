// app.js — Seeds of Truth UI logic (Flask routes + password gate)

'use strict';

/* =========================================================
   0) TUNABLES / CONSTANTS
   ========================================================= */
const CFG = {
  // LocalStorage keys
  LS_THEME: 'sot-theme',
  LS_SIDEBAR_COLLAPSED: 'sot-sidebar-collapsed',
  LS_TOOLS: 'sot-tools',
  LS_SAVED_CONVOS: 'sot-saved-conversations-v1',
  LS_FEEDBACK: 'sot-feedback-v1',

  // Limits
  MAX_CONTEXT_TURNS: 5,
  MAX_CONVO_TURNS: 50,
  MAX_SAVED_CONVOS: 25,
  MAX_FEEDBACK_ITEMS: 200,
  MAX_REFS: 10,

  // UI behavior
  DEFAULT_THEME: 'light',      // 'light' | 'dark'
  DEFAULT_HISTORY_TURNS: 2,    // 0..5
  DEFAULT_MODE: 'chat',        // 'search' | 'chat' | 'ab'
  TEXTAREA_MAX_HEIGHT: 140,    // px

  // Polling
  STATUS_POLL_MS: 15000,
  QUEUE_POLL_MS: 15000,

  // Flask endpoints
  API: {
    UNLOCK: '/api/unlock',
    ACCESS: '/api/access',
    SEARCH: '/api/search',
    CHAT: '/api/chat',
    AB: '/api/ab',
    FEEDBACK: '/api/feedback',
    STATUS: '/api/status',
    QUEUE: '/api/queue',
    PING: '/api/ping'
  }
};

/* =========================================================
   1) DOM LOOKUPS (set in init)
   ========================================================= */
const els = {}; // populated in initDom()

/* =========================================================
   2) STATE
   ========================================================= */
const toolState = {
  historyTurns: CFG.DEFAULT_HISTORY_TURNS,
  mode: CFG.DEFAULT_MODE
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
function clampInt(n, min, max, fallback) {
  const x = parseInt(n, 10);
  if (Number.isNaN(x)) return fallback;
  return Math.max(min, Math.min(max, x));
}

function clamp1to10(n) {
  const x = parseInt(n, 10);
  if (Number.isNaN(x)) return null;
  return Math.max(1, Math.min(10, x));
}

function safeJsonParse(str, fallback) {
  try { return JSON.parse(str); } catch (_) { return fallback; }
}

function nowId(prefix) {
  return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function formatDuration(totalSeconds) {
  const s = Math.max(0, totalSeconds | 0);
  const m = Math.floor(s / 60);
  const r = s % 60;
  if (m <= 0) return `${r}s`;
  return `${m}m ${r}s`;
}

/* =========================================================
   4) MODAL (custom alert/confirm)
   ========================================================= */
let modalResolve = null;

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

function onModalKeydown(e) {
  if (e.key === 'Escape') closeModal(false);
}

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

function modalConfirm({
  title = 'Confirm',
  message = 'Are you sure?',
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  danger = false
} = {}) {
  return new Promise((resolve) => {
    modalResolve = resolve;
    openModal({
      title,
      message,
      buttons: [
        { label: cancelText, value: false },
        { label: confirmText, value: true, variant: danger ? 'danger' : 'primary' }
      ]
    });
  });
}

function modalAlert({
  title = 'Notice',
  message = '',
  okText = 'OK'
} = {}) {
  return new Promise((resolve) => {
    modalResolve = resolve;
    openModal({
      title,
      message,
      buttons: [{ label: okText, value: true, variant: 'primary' }]
    });
  });
}

/* =========================================================
   5) THEME
   ========================================================= */
function applyTheme(mode) {
  if (mode === 'light') els.body.classList.add('light');
  else els.body.classList.remove('light');
  try { localStorage.setItem(CFG.LS_THEME, mode); } catch (_) {}
}

function initTheme() {
  let mode = CFG.DEFAULT_THEME;
  try {
    const stored = localStorage.getItem(CFG.LS_THEME);
    if (stored === 'light' || stored === 'dark') mode = stored;
  } catch (_) {}
  applyTheme(mode);

  els.themeToggleButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const isLightNow = els.body.classList.contains('light');
      applyTheme(isLightNow ? 'dark' : 'light');
    });
  });
}

/* =========================================================
   6) SIDEBAR COLLAPSE / MOBILE MENU
   ========================================================= */
function setSidebarCollapsed(collapsed) {
  els.body.classList.toggle('sidebar-collapsed', !!collapsed);
  try { localStorage.setItem(CFG.LS_SIDEBAR_COLLAPSED, collapsed ? '1' : '0'); } catch (_) {}
}

function initSidebarCollapse() {
  try {
    if (localStorage.getItem(CFG.LS_SIDEBAR_COLLAPSED) === '1') setSidebarCollapsed(true);
  } catch (_) {}

  if (els.sidebarCollapseBtn) els.sidebarCollapseBtn.addEventListener('click', () => setSidebarCollapsed(true));
  if (els.sidebarOpenBtn) els.sidebarOpenBtn.addEventListener('click', () => setSidebarCollapsed(false));
}

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
function loadToolState() {
  try {
    const raw = localStorage.getItem(CFG.LS_TOOLS);
    if (!raw) return;
    const parsed = safeJsonParse(raw, {});
    if (typeof parsed.historyTurns === 'number') {
      toolState.historyTurns = clampInt(parsed.historyTurns, 0, CFG.MAX_CONTEXT_TURNS, CFG.DEFAULT_HISTORY_TURNS);
    }
    if (['search','chat','ab'].includes(parsed.mode)) toolState.mode = parsed.mode;
  } catch (_) {}
}

function saveToolState() {
  try { localStorage.setItem(CFG.LS_TOOLS, JSON.stringify(toolState)); } catch (_) {}
}

function renderToolState() {
  if (els.historySlider) els.historySlider.value = String(toolState.historyTurns);
  if (els.historyValue) els.historyValue.textContent = String(toolState.historyTurns);
  if (els.historyHelpN) els.historyHelpN.textContent = String(toolState.historyTurns);

  // if locked, force search
  if (!isUnlocked && (toolState.mode === 'chat' || toolState.mode === 'ab')) {
    toolState.mode = 'search';
  }

  const id =
    toolState.mode === 'search' ? 'mode-search' :
    toolState.mode === 'ab'     ? 'mode-ab' :
                                  'mode-chat';
  const el = document.getElementById(id);
  if (el) el.checked = true;

  if (els.modeHelp) {
    els.modeHelp.textContent =
      toolState.mode === 'search' ? 'Search the corpus without AI' :
      toolState.mode === 'ab'     ? 'A/B test two responses and select the best one' :
                                    'AI chat: normal chat mode';
  }
  
  if (els.referencesTitle) {
    els.referencesTitle.textContent = (toolState.mode === 'search') ? 'Search Results' : 'References';
  }

}

function initToolsPopup() {
  if (!els.toolsBtn || !els.toolsPopup) return;

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
      toolState.historyTurns = clampInt(els.historySlider.value, 0, CFG.MAX_CONTEXT_TURNS, CFG.DEFAULT_HISTORY_TURNS);
      renderToolState();
      saveToolState();
    });
  }

  // radios change
  if (els.modeRadios && els.modeRadios.length) {
    els.modeRadios.forEach(r => {
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
}

/* =========================================================
   8) STATUS PANEL
   ========================================================= */
function setEndpointStatus(status) {
  if (!els.endpointDot) return;

  els.endpointDot.classList.remove('red','yellow','green');

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

function setQueueStatus(queriesInLine) {
  const q = Math.max(0, parseInt(queriesInLine, 10) || 0);
  if (els.queueCountEl) els.queueCountEl.textContent = String(q);
  if (els.queueEtaEl) els.queueEtaEl.textContent = formatDuration(q * 45);
}

function pushStatusMessage(text) {
  const msg = String(text || '').trim();
  if (!msg || !els.statusMessagesEl) return;

  const placeholder = els.statusMessagesEl.querySelector('.status-message.muted');
  if (placeholder) placeholder.remove();

  const el = document.createElement('div');
  el.className = 'status-message';
  const ts = new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
  el.textContent = `[${ts}] ${msg}`;
  els.statusMessagesEl.prepend(el);

  const items = els.statusMessagesEl.querySelectorAll('.status-message');
  if (items.length > 1) items[items.length - 1].remove();
}

/* =========================================================
   9) FEEDBACK MODAL (local save for now)
   ========================================================= */
function showToast() {
  if (!els.fbToast) return;
  els.fbToast.classList.add('show');
  els.fbToast.setAttribute('aria-hidden', 'false');
  setTimeout(() => {
    els.fbToast.classList.remove('show');
    els.fbToast.setAttribute('aria-hidden', 'true');
  }, 1400);
}

function openFeedbackModal(target) {
  const isRef = target?.type === 'reference';
  feedbackTarget = target;

  // reset
  if (els.fbAccuracy) els.fbAccuracy.value = 8;
  if (els.fbStyle) els.fbStyle.value = 8;
  if (els.fbRelevance) els.fbRelevance.value = 8;
  if (els.fbComments) els.fbComments.value = '';

  if (els.fbFieldAccuracy) els.fbFieldAccuracy.style.display = isRef ? 'none' : '';
  if (els.fbFieldStyle) els.fbFieldStyle.style.display = isRef ? 'none' : '';

  const label = isRef ? 'Reference' : 'Response';
  const snip = (target?.snippet || '').trim().replace(/\s+/g, ' ');
  const short = snip.length > 120 ? snip.slice(0, 120) + '…' : snip;
  if (els.fbMeta) els.fbMeta.textContent = `${label} ID: ${target?.id || 'n/a'}${short ? ' — ' + short : ''}`;

  els.fbOverlay.classList.add('show');
  els.fbOverlay.setAttribute('aria-hidden', 'false');

  (isRef ? els.fbRelevance : els.fbAccuracy)?.focus?.();
}

function closeFeedbackModal() {
  if (!els.fbOverlay) return;
  els.fbOverlay.classList.remove('show');
  els.fbOverlay.setAttribute('aria-hidden', 'true');
  feedbackTarget = null;
}

function saveFeedbackLocally(payload) {
  try {
    const arr = safeJsonParse(localStorage.getItem(CFG.LS_FEEDBACK) || '[]', []);
    arr.unshift(payload);
    if (arr.length > CFG.MAX_FEEDBACK_ITEMS) arr.length = CFG.MAX_FEEDBACK_ITEMS;
    localStorage.setItem(CFG.LS_FEEDBACK, JSON.stringify(arr));
  } catch (_) {}
}

async function submitFeedback() {
  if (!feedbackTarget) return;

  const isRef = feedbackTarget?.type === 'reference';

  const relevance = clamp1to10(els.fbRelevance.value);
  const accuracy = isRef ? undefined : clamp1to10(els.fbAccuracy.value);
  const style = isRef ? undefined : clamp1to10(els.fbStyle.value);

  const payload = {
    target: {
      type: feedbackTarget?.type || 'unknown',
      id: feedbackTarget?.id || null,
      snippet: feedbackTarget?.snippet || ''
    },
    ratings: {
      relevance,
      ...(isRef ? {} : { accuracy, style })
    },
    comments: (els.fbComments.value || '').trim(),
    createdAt: Date.now()
  };

  if (!relevance) {
    await modalAlert({ title: 'Missing rating', message: 'Please enter 1–10 for Relevance.' });
    return;
  }
  if (!isRef && (!accuracy || !style)) {
    await modalAlert({ title: 'Missing ratings', message: 'Please enter 1–10 for Accuracy and Style.' });
    return;
  }

  // Keep local for now
  saveFeedbackLocally(payload);

  closeFeedbackModal();
  showToast();
}

function initFeedbackModal() {
  if (els.fbClose) els.fbClose.addEventListener('click', closeFeedbackModal);
  if (els.fbCancel) els.fbCancel.addEventListener('click', closeFeedbackModal);
  if (els.fbSubmit) els.fbSubmit.addEventListener('click', submitFeedback);

  if (els.fbOverlay) {
    els.fbOverlay.addEventListener('click', (e) => {
      if (e.target === els.fbOverlay) closeFeedbackModal();
    });
    document.addEventListener('keydown', (e) => {
      if (els.fbOverlay.classList.contains('show') && e.key === 'Escape') closeFeedbackModal();
    });
  }
}

/* =========================================================
   10) CHAT UI RENDERING
   ========================================================= */
function scrollChatToBottom() {
  if (els.chatContainer) els.chatContainer.scrollTop = els.chatContainer.scrollHeight;
}

function setBotAvatar(el) {
  el.textContent = ""; // remove any text
  el.innerHTML = `
    <img src="/static/img/logo512.png" alt="" class="message-avatar-img" />
  `;
}

function appendMessage(text, role) {
  const row = document.createElement('div');
  row.className = 'message-row ' + (role === 'bot' ? 'bot' : 'user');

  const inner = document.createElement('div');
  inner.className = 'message-content';

  const avatar = document.createElement('div');
  avatar.className = 'message-avatar ' + (role === 'bot' ? 'bot' : 'user');
  if (role === "bot") {
    setBotAvatar(avatar);
  } else {
    avatar.textContent = "You";
  }

  const textEl = document.createElement('div');
  textEl.className = 'message-text';
  textEl.textContent = text;

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
      openFeedbackModal({ type: 'response', id: msgId, snippet: text });
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

// A/B mode: side-by-side answers with selectable choice + feedback
function appendABMessage(aText, bText, meta = {}) {
  const row = document.createElement('div');
  row.className = 'message-row bot';

  const inner = document.createElement('div');
  inner.className = 'message-content';

  const avatar = document.createElement('div');
  avatar.className = 'message-avatar bot';
  setBotAvatar(avatar);

  const wrap = document.createElement('div');
  wrap.className = 'ab-wrap';

  function selectPanel(panel, variant) {
    wrap.querySelectorAll('.ab-panel').forEach(p => {
      p.classList.remove('selected');
      const btn = p.querySelector('.ab-select-btn');
      if (btn) btn.innerHTML = 'Select this response';
    });

    panel.classList.add('selected');

    const btn = panel.querySelector('.ab-select-btn');
    if (btn) btn.innerHTML = '✓ Selected';
  }

  function makePanel(label, text, variant) {
    const panel = document.createElement('div');
    panel.className = 'ab-panel';

    // header
    const head = document.createElement('div');
    head.className = 'ab-head';

    const lbl = document.createElement('div');
    lbl.className = 'ab-label';
    lbl.textContent = label;

    const actions = document.createElement('div');
    actions.className = 'ab-actions';

    const id = `ab_${variant}_${nowId('msg')}`;

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
      openFeedbackModal({ type: 'response', id, snippet: `[A/B ${variant}] ${text}` });
    });

    actions.appendChild(commentBtn);
    head.appendChild(lbl);
    head.appendChild(actions);

    // body
    const body = document.createElement('div');
    body.className = 'message-text';
    body.textContent = text;

    // footer
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
    return panel;
  }

  wrap.appendChild(makePanel(meta.labelA || 'Response A', aText, 'A'));
  wrap.appendChild(makePanel(meta.labelB || 'Response B', bText, 'B'));

  inner.appendChild(avatar);
  inner.appendChild(wrap);
  row.appendChild(inner);
  els.messagesEl.appendChild(row);
  scrollChatToBottom();
}

/* =========================================================
   11) REFERENCES
   ========================================================= */
function linkifyPlainUrls(htmlStr) {
  // Turn bare URLs into <a> links, but avoid touching existing tags too much.
  // This is intentionally conservative: it won't linkify inside existing attributes.
  const urlRe = /(^|[\s>])(https?:\/\/[^\s<]+)/g;
  return String(htmlStr || '').replace(urlRe, (m, prefix, url) => {
    return `${prefix}<a href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>`;
  });
}

function toHttpsUrl(u) {
  const s = String(u || "").trim();
  if (!s) return "";
  if (/^https?:\/\//i.test(s)) return s;
  return "https://" + s.replace(/^\/+/, "");
}

function looksLikeUrl(s) {
  const t = String(s || "").trim();
  if (!t) return false;
  // allow full URLs
  if (/^https?:\/\//i.test(t)) return true;
  // allow domains/paths like www.wanttoknow.info/...
  return /^[a-z0-9.-]+\.[a-z]{2,}(\/|$)/i.test(t);
}

function getRefLink(ref) {
  // Prefer explicit url, otherwise fall back to source if it looks like a URL
  const u = String(ref?.url || "").trim();
  if (u) return u;

  const s = String(ref?.source || "").trim();
  if (looksLikeUrl(s)) return s;

  return "";
}


function sanitizeBasicHtml(htmlStr) {
  // Minimal sanitizer (NOT bulletproof). Best practice is DOMPurify.
  // Removes scripts/styles/iframes and strips on* handlers.
  const tpl = document.createElement('template');
  tpl.innerHTML = String(htmlStr || '');

  // Remove dangerous elements
  tpl.content.querySelectorAll('script, style, iframe, object, embed, link, meta').forEach(n => n.remove());

  // Strip inline event handlers and javascript: URLs
  tpl.content.querySelectorAll('*').forEach(el => {
    [...el.attributes].forEach(attr => {
      const name = attr.name.toLowerCase();
      const val = String(attr.value || '').trim().toLowerCase();
      if (name.startsWith('on')) el.removeAttribute(attr.name);
      if ((name === 'href' || name === 'src') && val.startsWith('javascript:')) el.removeAttribute(attr.name);
    });

    // Force safe link behavior
    if (el.tagName === 'A') {
      el.setAttribute('target', '_blank');
      el.setAttribute('rel', 'noopener noreferrer');
    }
  });

  return tpl.innerHTML;
}

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
  nodesToClearAfter.forEach(n => { n.nodeValue = ''; });

  // Cleanup: remove now-empty elements to avoid lots of blank tags
  containerEl.querySelectorAll('*').forEach(el => {
    if (!el.textContent.trim() && !el.querySelector('img, br, hr')) {
      el.remove();
    }
  });
}
   
   
function setReferences(refs) {
  const list = Array.isArray(refs) ? refs.slice(0, CFG.MAX_REFS) : [];
  els.referencesContainer.innerHTML = '';

  if (!list.length) {
    els.referencesCount.textContent = '0 items';
    els.referencesEmpty.style.display = 'block';
    return;
  }

  els.referencesEmpty.style.display = 'none';
  els.referencesCount.textContent = list.length + ' item' + (list.length === 1 ? '' : 's');

  list.forEach((ref, idx) => {
	console.log("REF", idx, Object.keys(ref), ref.url, ref.source, ref);  
    const card = document.createElement('article');
    card.className = 'ref-card';

    const refId = `ref_${idx}_${Math.random().toString(16).slice(2)}`;

    const header = document.createElement('div');
    header.className = 'ref-header';

    const right = document.createElement('div');
    right.className = 'ref-right';

    // ✅ clickable URL (green, top-right)
	const linkText = getRefLink(ref);
	const href = toHttpsUrl(linkText);

	const badge = document.createElement(linkText ? 'a' : 'span');
	badge.className = 'ref-badge';
	badge.textContent = ref.source || 'Corpus';

	if (linkText) {
	  badge.href = href;
	  badge.target = '_blank';
	  badge.rel = 'noopener noreferrer';
	  badge.addEventListener("click", (e) => e.stopPropagation());
	}

	if (ref.url) {
	  const href = toHttpsUrl(String(ref.url).trim());
	  badge.href = href;
	  badge.target = '_blank';
	  badge.rel = 'noopener noreferrer';

	  // keep parent handlers from hijacking the click
	  badge.addEventListener('click', (e) => {
		e.stopPropagation();
	  });

	  // If something upstream still blocks navigation, use this instead:
	  // badge.addEventListener('click', (e) => {
	  //   e.stopPropagation();
	  //   e.preventDefault();
	  //   window.open(href, "_blank", "noopener,noreferrer");
	  // });
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
      openFeedbackModal({
        type: 'reference',
        id: refId,
        snippet: (ref.title ? ref.title + ' — ' : '') + (ref.snippet || '')
      });
    });

    right.appendChild(badge);
    right.appendChild(cbtn);

    header.appendChild(right);

    const snippetEl = document.createElement('div');
    snippetEl.className = 'ref-snippet';

    // Prefer server-rendered HTML, fall back to plain snippet/text
    const raw = (ref.snippet ?? ref.text ?? ref.content ?? '');
    const html1 = linkifyPlainUrls(raw);
    const html2 = sanitizeBasicHtml(html1);

    snippetEl.innerHTML = html2;

    // limit to 500 words displayed
    truncateElementToWords(snippetEl, 500);

    card.appendChild(header);
    card.appendChild(snippetEl);

    els.referencesContainer.appendChild(card);
  });
}


/* =========================================================
   12) CONVERSATION HISTORY (client-side)
   ========================================================= */
function pushTurn(userText, assistantText) {
  convoTurns.push({ user: userText, assistant: assistantText });
  if (convoTurns.length > CFG.MAX_CONVO_TURNS) convoTurns.shift();
}

function getContextTurns(n) {
  const turns = clampInt(n, 0, CFG.MAX_CONTEXT_TURNS, CFG.DEFAULT_HISTORY_TURNS);
  return convoTurns.slice(-turns);
}

/* =========================================================
   13) SAVED CONVERSATIONS (localStorage)
   ========================================================= */
function loadSavedConversations() {
  const raw = localStorage.getItem(CFG.LS_SAVED_CONVOS);
  const parsed = safeJsonParse(raw || '[]', []);
  return Array.isArray(parsed) ? parsed : [];
}

function saveSavedConversations(list) {
  try { localStorage.setItem(CFG.LS_SAVED_CONVOS, JSON.stringify(list)); } catch (_) {}
}

function makeConversationTitle(turns) {
  const first = turns?.find(t => t?.user)?.user || 'Conversation';
  const t = String(first).trim().replace(/\s+/g, ' ');
  return t.length > 42 ? t.slice(0, 42) + '…' : t;
}

function makeConversationPreview(turns) {
  if (!Array.isArray(turns) || turns.length === 0) return '';
  const last = turns[turns.length - 1];
  const txt = String(last.assistant || last.user || '').trim().replace(/\s+/g, ' ');
  return txt.length > 60 ? txt.slice(0, 60) + '…' : txt;
}

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

  saved.slice()
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
    turns: convoTurns.slice()
  };

  saved.unshift(item);
  if (saved.length > CFG.MAX_SAVED_CONVOS) saved.length = CFG.MAX_SAVED_CONVOS;

  saveSavedConversations(saved);
  renderRecentList();
  pushStatusMessage(`Saved conversation: "${item.title}"`);
}

function loadConversationById(id) {
  const saved = loadSavedConversations();
  const item = saved.find(x => x.id === id);
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

async function deleteConversationById(id) {
  const ok = await modalConfirm({
    title: 'Delete saved conversation?',
    message: 'This will remove the saved conversation from this browser. This cannot be undone.',
    confirmText: 'Delete',
    cancelText: 'Cancel',
    danger: true
  });
  if (!ok) return;

  const saved = loadSavedConversations().filter(c => c.id !== id);
  saveSavedConversations(saved);
  renderRecentList();
  pushStatusMessage('Conversation deleted.');
}

/* =========================================================
   14) INPUT AREA
   ========================================================= */
function autoResizeTextarea() {
  if (!els.chatInput) return;
  els.chatInput.style.height = 'auto';
  els.chatInput.style.height = Math.min(els.chatInput.scrollHeight, CFG.TEXTAREA_MAX_HEIGHT) + 'px';
  els.chatInput.style.overflowY = (els.chatInput.scrollHeight > CFG.TEXTAREA_MAX_HEIGHT) ? 'auto' : 'hidden';
}

async function clearCurrentChat() {
  const hasMessages = (convoTurns && convoTurns.length) || (els.messagesEl && els.messagesEl.children.length);
  if (!hasMessages) {
    await modalAlert({ title: 'Nothing to clear', message: 'There are no messages in the current chat yet.' });
    return;
  }

  const ok = await modalConfirm({
    title: 'Clear current chat?',
    message: 'This clears the current chat on this page. Saved chats are not affected.',
    confirmText: 'Clear',
    cancelText: 'Cancel',
    danger: true
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
async function apiPost(url, payload) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'same-origin', // ✅ required for Flask session cookie
    body: JSON.stringify(payload || {})
  });

  let data = {};
  try { data = await res.json(); } catch (_) {}

  if (!res.ok) {
    const msg = data?.message || data?.error || `Request failed (${res.status})`;
    const err = new Error(msg);
    err.status = res.status;
    err.data = data;
    throw err;
  }
  return data;
}

async function apiGet(url) {
  const res = await fetch(url, { method: 'GET', credentials: 'same-origin' });
  let data = {};
  try { data = await res.json(); } catch (_) {}
  if (!res.ok) throw new Error(data?.message || `Request failed (${res.status})`);
  return data;
}

async function apiSearch(payload) {
  const res = await fetch('/api/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'same-origin',
    body: JSON.stringify(payload)
  });

  const text = await res.text();

  console.log("[/api/search] status:", res.status);
  console.log("[/api/search] raw body:", text.slice(0, 2000));

  let data = {};
  try { data = text ? JSON.parse(text) : {}; } catch (_) {}

  console.log("[/api/search] parsed keys:", Object.keys(data || {}));
  console.log("[/api/search] parsed data:", data);

  // ALSO treat ok:false as an error even if HTTP 200
  if (!res.ok || data.ok === false) {
    const msg = data.error || data.message || text || `HTTP ${res.status}`;
    throw new Error(`apiSearch failed: ${res.status} — ${msg}`);
  }

  return data;
}



async function apiChat(payload) {
  return apiPost(CFG.API.CHAT, payload);
}

async function apiAB(payload) {
  return apiPost(CFG.API.AB, payload);
}

async function apiUnlock(password) {
  return apiPost(CFG.API.UNLOCK, { password });
}

async function apiAccess() {
  return apiGet(CFG.API.ACCESS);
}

async function apiStatus() {
  return apiGet(CFG.API.STATUS);
}

async function apiQueue() {
  return apiGet(CFG.API.QUEUE);
}

/* =========================================================
   16) CHAT HANDLING (submit + render)
   ========================================================= */
function getUserId() {
  const KEY = 'sot_user_id';
  let id = localStorage.getItem(KEY);
  if (!id) {
    // Stable per-browser ID (good enough for dev + non-auth chat)
    id = (crypto?.randomUUID ? crypto.randomUUID() : `u_${Date.now()}_${Math.random().toString(16).slice(2)}`);
    localStorage.setItem(KEY, id);
  }
  return id;
}   
   
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

  const payload = {
    user_id: userId,          
    message: text,
    mode: toolState.mode,
    history_turns: toolState.historyTurns,
    context: contextTurns
  };

  try {
    // Enforce gate client-side (server enforces too)
    if (!isUnlocked && (toolState.mode === 'chat' || toolState.mode === 'ab')) {
      toolState.mode = 'search';
      renderToolState();
      saveToolState();
    }

    if (toolState.mode === 'search') {
      const payload = {
        user_id: userId,      
        query: text,
        max_n: CFG.MAX_REFS,
        mode: 'search',
        history_turns: toolState.historyTurns,
        context: contextTurns,
      };

      console.log('[Search → Flask] payload:', payload);

      try {
        const data = await apiSearch(payload);
        console.log('[Search ← Flask] response:', data);

        const bot = data.message || `Found ${data.num_results ?? (data.results?.length ?? 0)} items. Results below`;
        appendMessage(bot, 'bot');
        pushTurn(text, bot);

        const refs = (Array.isArray(data.results) ? data.results : [])
          .slice(0, CFG.MAX_REFS)
          .map((r) => ({
            title: r.title || r.source_title || 'Reference',
            source: r.source || r.publication || 'Corpus',
            snippet: r.snippet || r.text || r.excerpt || '',
            url: r.url || r.link || ''
          }));

        setReferences(refs);
      } catch (err) {
        console.error('[Search] apiSearch failed:', err);
        appendMessage('Search request failed (dev). Check server logs.', 'bot');
        pushStatusMessage(String(err?.message || err));
        setReferences([]);
      }
      return;
    }

    if (toolState.mode === 'ab') {
      // payload already has user_id
      const data = await apiAB(payload);
      appendABMessage(data.a || '', data.b || '', { labelA: data.labelA, labelB: data.labelB });
      pushTurn(text, `Response A:\n${data.a || ''}\n\nResponse B:\n${data.b || ''}`);
      setReferences(Array.isArray(data.references) ? data.references : []);
      return;
    }

    // chat
    const data = await apiChat(payload);  // payload already has user_id
    const reply = data.reply || data.message || data.status || '';
    appendMessage(reply || '(no reply)', 'bot');
    pushTurn(text, reply || '(no reply)');
    setReferences(Array.isArray(data.references) ? data.references : []);

  } catch (err) {
    if (err?.status === 403) {
      setModeAccess(false);
      toolState.mode = 'search';
      renderToolState();
      saveToolState();
      appendMessage('Not today. Search mode only.', 'bot');
      pushTurn(text, 'Not today. Search mode only.');
      return;
    }

    appendMessage('Error contacting server. Please try again.', 'bot');
    pushStatusMessage(String(err?.message || err));
  }
}


/* =========================================================
   17) ABOUT MODAL and PING TEST
   ========================================================= */
function initAboutModal() {
  const btn = document.getElementById("about-btn");
  const overlay = document.getElementById("about-overlay");
  const closeBtn = document.getElementById("about-close");

  if (!btn || !overlay || !closeBtn) return;

  function open() {
    overlay.classList.add("show");
    overlay.setAttribute("aria-hidden", "false");
  }

  function close() {
    overlay.classList.remove("show");
    overlay.setAttribute("aria-hidden", "true");
  }

  btn.addEventListener("click", open);
  closeBtn.addEventListener("click", close);

  // click outside
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) close();
  });

  // escape key
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && overlay.classList.contains("show")) close();
  });
}

function initPingTest() {
  const btn = document.getElementById("ping-btn");
  const output = document.getElementById("ping-result");
  if (!btn || !output) return;

  btn.addEventListener("click", async () => {
    output.textContent = "Sending request...";

    try {
      const data = await apiPost(CFG.API.PING, {
        from: "browser",
        test: "JS → Flask → JS"
      });
      output.textContent = JSON.stringify(data, null, 2);
    } catch (err) {
      output.textContent = "Error: " + (err?.message || String(err));
    }
  });
}

/* =========================================================
   18) RANGE FILL + AUTOSIZE (helpers)
   ========================================================= */
function initRangeFill() {
  const r = document.getElementById('history-slider');
  if (!r) return;

  function setFill(){
    const min = Number(r.min || 0);
    const max = Number(r.max || 100);
    const val = Number(r.value || 0);
    const pct = ((val - min) / (max - min)) * 100;
    r.style.setProperty('--fill', pct + '%');
  }

  r.addEventListener('input', setFill);
  setFill();
}

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

function closeLockModal() {
  const lockModal = document.getElementById("lock-modal");
  const lockPass  = document.getElementById("lock-pass");
  const lockMsg   = document.getElementById("lock-msg");

  if (!lockModal) return;

  lockModal.classList.remove("open");
  lockModal.setAttribute("aria-hidden", "true");

  if (lockMsg) lockMsg.textContent = "";
  if (lockPass) lockPass.value = "";
}
   
function setModeAccess(unlocked) {
  isUnlocked = !!unlocked;

  const modeSearch = document.getElementById("mode-search");
  const modeChat = document.getElementById("mode-chat");
  const modeAb = document.getElementById("mode-ab");
  const lockBtnEl = document.getElementById("lock-btn");

  if (modeChat) modeChat.disabled = !isUnlocked;
  if (modeAb) modeAb.disabled = !isUnlocked;

  if (!isUnlocked) toolState.mode = "search";
  if (!isUnlocked && modeSearch) modeSearch.checked = true;

  if (lockBtnEl) {
    lockBtnEl.classList.toggle("unlocked", isUnlocked);
    lockBtnEl.disabled = isUnlocked;
    lockBtnEl.setAttribute(
      "aria-label",
      isUnlocked ? "Access granted. Connecting..." : "Restricted access"
    );
  }

  renderToolState();
  saveToolState();

  // close after any re-render, using global re-querying closer
  if (isUnlocked) closeLockModal();
}

function initLockUI() {
  const lockBtn = document.getElementById("lock-btn");
  const lockModal = document.getElementById("lock-modal");
  const lockClose = document.getElementById("lock-close");
  const lockEnter = document.getElementById("lock-enter");
  const lockPass = document.getElementById("lock-pass");
  const lockMsg = document.getElementById("lock-msg");

  if (!lockBtn || !lockModal || !lockClose || !lockEnter || !lockPass || !lockMsg) return;

  function openLockModal() {
    lockModal.classList.add("open");
    lockModal.setAttribute("aria-hidden", "false");
    lockMsg.textContent = "";
    lockPass.value = "";
    setTimeout(() => lockPass.focus(), 0);
  }

  async function tryUnlock() {
    const pw = (lockPass.value || "").trim();
    if (!pw) return;

    lockEnter.disabled = true;
    lockPass.disabled = true;
    lockMsg.textContent = "Checking…";

    try {
      const data = await apiUnlock(pw);
      const unlocked = !!(data && data.unlocked === true);

	  if (unlocked) {
	    setModeAccess(true);
	  } else {
	    setModeAccess(false);
	    lockMsg.textContent = (data && (data.message || data.error)) || "Incorrect password";
	  }

    } catch (err) {
      setModeAccess(false);
      lockMsg.textContent = err?.message || "Not today";
    } finally {
      lockEnter.disabled = false;
      lockPass.disabled = false;
    }
  }

  lockBtn.addEventListener("click", openLockModal);
  lockClose.addEventListener("click", closeLockModal);
  lockModal.addEventListener("click", (e) => { if (e.target === lockModal) closeLockModal(); });
  lockEnter.addEventListener("click", tryUnlock);

  lockPass.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();      // prevents newline / implicit form submit
      tryUnlock();
    }
    if (e.key === "Escape") closeLockModal();
  });
}


/* =========================================================
   20) STATUS + QUEUE POLLING
   ========================================================= */
async function refreshStatusOnce() {
  try {
    const data = await apiStatus();
    // If your /api/status includes these fields, reflect them:
    // { unlocked: bool, retrieval_state_ready: bool, ... }
    if (typeof data?.unlocked === 'boolean') setModeAccess(data.unlocked);

    if (data?.retrieval_state_ready === true) setEndpointStatus('ready');
    else setEndpointStatus('starting');
  } catch (_) {
    setEndpointStatus('off');
  }
}

async function refreshQueueOnce() {
  try {
    const data = await apiQueue();
    // expected: { ok:true, queries_in_line: N } or similar
    const q = data?.queries_in_line ?? data?.queue ?? 0;
    setQueueStatus(q);
  } catch (_) {
    setQueueStatus(0);
  }
}

function startPolling() {
  refreshStatusOnce();
  refreshQueueOnce();

  setInterval(refreshStatusOnce, CFG.STATUS_POLL_MS);
  setInterval(refreshQueueOnce, CFG.QUEUE_POLL_MS);
}

/* =========================================================
   21) INIT / WIRING (bottom)
   ========================================================= */
function initDom() {
  els.body = document.body;

  // core chat
  els.chatForm = document.getElementById('chat-form');
  els.chatInput = document.getElementById('chat-input');
  els.messagesEl = document.getElementById('messages');
  els.chatContainer = document.getElementById('chat-container');
  els.welcomeMessage = document.getElementById('welcome-message');

  // theme + sidebar
  els.themeToggleButtons = Array.from(document.querySelectorAll('[data-theme-toggle]'));
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

  // custom modal
  els.modalOverlay = document.getElementById('modal-overlay');
  els.modalTitle = document.getElementById('modal-title');
  els.modalMessage = document.getElementById('modal-message');
  els.modalActions = document.getElementById('modal-actions');
  els.modalCloseBtn = document.getElementById('modal-close');
}

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
    try { els.chatInput.setSelectionRange(els.chatInput.value.length, els.chatInput.value.length); } catch (_) {}
  }

  // save convo
  if (els.saveConvoBtn) els.saveConvoBtn.addEventListener('click', saveCurrentConversation);

  // trash chat
  if (els.trashChatBtn) els.trashChatBtn.addEventListener('click', clearCurrentChat);
}

function init() {
  initDom();

  // sanity
  if (!els.chatForm || !els.chatInput || !els.messagesEl) {
    console.warn('Seeds of Truth app.js: required chat elements not found.');
    return;
  }

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

  // ask server whether this session is already unlocked
  apiAccess().then(d => setModeAccess(!!d.unlocked)).catch(() => setModeAccess(false));

  // polling for status/queue
  startPolling();

  // expose debug hooks
  window.setReferences = setReferences;
  window.openFeedbackModal = openFeedbackModal;
  window.pushStatusMessage = pushStatusMessage;
}

document.addEventListener('DOMContentLoaded', init);

(function clickDistortion(){
  const host = document.getElementById('click-distort');
  const svg = document.querySelector('filter#sot-displace');
  if (!host || !svg) return;

  // Find the displacement map inside the filter
  const disp = document.querySelector('#sot-displace feDisplacementMap');
  const turb = document.querySelector('#sot-displace feTurbulence');
  if (!disp || !turb) return;

  let seed = 2;

  function spawn(x, y){
    // Lens
    const lens = document.createElement('div');
    lens.className = 'click-lens';
    lens.style.left = x + 'px';
    lens.style.top  = y + 'px';

    // Ring
    const ring = document.createElement('div');
    ring.className = 'click-ring';
    ring.style.left = x + 'px';
    ring.style.top  = y + 'px';

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
  window.addEventListener('pointerdown', (e) => {
    // ignore right-click
    if (e.button === 2) return;
    spawn(e.clientX, e.clientY);
  }, { passive: true });
})();

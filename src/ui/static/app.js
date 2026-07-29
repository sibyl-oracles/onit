/* OnIt web UI — vanilla JS single-page app.
 * Talks to the FastAPI backend in src/ui/api.py:
 *   GET  /api/config /api/history /api/sessions /api/logs
 *   POST /api/chat (SSE) /api/chat/stop /api/clear /api/upload /api/sessions/new
 * SSE events: token, phase_end, status, answer_start, done, error.
 */
(function () {
  "use strict";

  // ── State ──────────────────────────────────────────────────────
  // ?session=<uuid> deep-links a specific chat; falls back to the last one used
  const urlSession = new URLSearchParams(location.search).get("session");
  const state = {
    config: null,
    sessionId: urlSession || localStorage.getItem("onit.sid") || null,
    processing: false,
    attachments: [],       // [{name, url}]
    userScrolledUp: false,
    logsOpen: false,
    logsTimer: null,
    pollTimer: null,
  };

  const $ = (id) => document.getElementById(id);
  const el = {
    app: $("app"), login: $("login"),
    sidebar: $("sidebar"), sessionList: $("session-list"),
    messages: $("messages"), welcome: $("welcome"),
    chatScroll: $("chat-scroll"),
    input: $("input"), sendBtn: $("send-btn"), stopBtn: $("stop-btn"),
    attachBtn: $("attach-btn"), fileInput: $("file-input"),
    attachments: $("attachments"),
    newChat: $("new-chat"), clearChat: $("clear-chat"), clearAll: $("clear-all"),
    themeToggle: $("theme-toggle"),
    logsToggle: $("logs-toggle"), logsDrawer: $("logs-drawer"),
    logsBody: $("logs-body"), logsClose: $("logs-close"),
    sidebarOpen: $("sidebar-open"), sidebarClose: $("sidebar-close"),
    sidebarFoot: $("sidebar-foot"), userEmail: $("user-email"),
    userAvatar: $("user-avatar"),
    brandTitle: $("brand-title"), welcomeTitle: $("welcome-title"),
    loginTitle: $("login-title"), composerHint: $("composer-hint"),
  };

  // ── Markdown rendering ─────────────────────────────────────────
  marked.setOptions({ gfm: true, breaks: true });

  function renderMarkdown(container, text) {
    const html = DOMPurify.sanitize(marked.parse(text || ""), {
      ADD_ATTR: ["target"],
    });
    container.innerHTML = html;
    container.querySelectorAll("a[href]").forEach((a) => {
      const href = a.getAttribute("href") || "";
      if (href.startsWith("/uploads/")) {
        a.setAttribute("download", "");
      } else {
        vetLink(a, href);
      }
    });
    // Images the agent made and referenced itself: ask for the inline
    // rendering of the upload route, since the plain URL is a download.
    container.querySelectorAll("img[src]").forEach((img) => {
      const src = img.getAttribute("src") || "";
      if (!src.startsWith("/uploads/")) return;
      img.setAttribute("src", inlineUrl(src));
      img.classList.add("msg-image");
      img.setAttribute("loading", "lazy");
    });
    container.querySelectorAll("pre > code").forEach((code) => {
      try { hljs.highlightElement(code); } catch (e) { /* ignore */ }
      decorateCodeBlock(code);
    });
  }

  // ── Link verification ──────────────────────────────────────────
  // The agent sometimes emits malformed or hallucinated URLs
  // (e.g. https://ge.php, https://manual). External links stay
  // non-clickable until POST /api/verify_links confirms they resolve;
  // failures render as plain text.
  //
  // mailto: links ride the same path, but the server answers them from the
  // session's sources rather than the network: an address is clickable when
  // it appeared in a document, a web search result, or the user's own
  // message, and struck through when only the model ever said it.
  const linkCache = new Map();   // url -> "ok" | "bad"
  const linkQueue = new Set();   // urls awaiting a verify request
  let linkFlushTimer = null;
  const VERIFY_BATCH = 20;       // must match _VERIFY_MAX_URLS server-side
  const EMAIL_RE = /^[^\s@]+@[^\s@.]+(\.[^\s@.]+)*\.[a-z]{2,}$/i;

  function urlShapeOk(href) {
    let u;
    try { u = new URL(href); } catch (e) { return false; }
    if (u.protocol === "mailto:") {
      // Screen the shape here so obvious junk never costs a round trip;
      // whether the address is *grounded* is the server's call.
      return EMAIL_RE.test(decodeURIComponent(u.pathname.split("?")[0]).trim());
    }
    if (u.protocol !== "http:" && u.protocol !== "https:") return false;
    return u.hostname.includes(".");
  }

  function delinkify(a, cls, title) {
    const span = document.createElement("span");
    span.className = cls;
    if (title) span.title = title;
    span.textContent = a.textContent;
    a.replaceWith(span);
    return span;
  }

  function isMailto(href) {
    return /^mailto:/i.test(href);
  }

  // Reason shown on hover, so a struck-through address doesn't read as a
  // dead link when it's really an ungrounded one.
  function brokenTitle(href) {
    return isMailto(href)
      ? "Address not found in any document or search result"
      : "Link could not be verified";
  }

  function pendingTitle(href) {
    return isMailto(href) ? "Checking sources…" : "Verifying link…";
  }

  function activateLink(a, href) {
    a.setAttribute("href", href);
    if (!isMailto(href)) {
      // A mail client handoff isn't a new browsing context; target/rel only
      // matter for the http(s) case.
      a.setAttribute("target", "_blank");
      a.setAttribute("rel", "noopener noreferrer");
    }
  }

  function vetLink(a, href) {
    if (!urlShapeOk(href)) {
      delinkify(a, "link-broken", brokenTitle(href));
      return;
    }
    const verdict = linkCache.get(href);
    if (verdict === "ok") {
      activateLink(a, href);
    } else if (verdict === "bad") {
      delinkify(a, "link-broken", brokenTitle(href));
    } else {
      const span = delinkify(a, "link-pending", pendingTitle(href));
      span.dataset.verifyUrl = href;
      queueLinkVerify(href);
    }
  }

  // http(s) verdicts are a property of the internet and survive a session
  // switch; email verdicts are a property of *this* session's sources, so
  // they must not leak into the next chat.
  function forgetEmailVerdicts() {
    for (const url of Array.from(linkCache.keys())) {
      if (isMailto(url)) linkCache.delete(url);
    }
  }

  function queueLinkVerify(url) {
    if (linkCache.has(url) || linkQueue.has(url)) return;
    linkQueue.add(url);
    clearTimeout(linkFlushTimer);
    linkFlushTimer = setTimeout(flushLinkQueue, 200);
  }

  async function flushLinkQueue() {
    while (linkQueue.size) {
      const urls = Array.from(linkQueue).slice(0, VERIFY_BATCH);
      urls.forEach((u) => {
        linkQueue.delete(u);
        linkCache.set(u, "pending");  // suppress duplicate requests mid-flight
      });
      let results = {};
      try {
        const res = await api("/api/verify_links", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ urls }),
        });
        results = (await res.json()).results || {};
      } catch (e) {
        // Verification unavailable: leave links pending (visible but
        // not clickable); a later render re-queues them.
        urls.forEach((u) => {
          if (linkCache.get(u) === "pending") linkCache.delete(u);
        });
        return;
      }
      for (const u of urls) {
        if (u in results) linkCache.set(u, results[u] ? "ok" : "bad");
        else if (linkCache.get(u) === "pending") linkCache.delete(u);
      }
      applyLinkVerdicts();
    }
  }

  function applyLinkVerdicts() {
    document.querySelectorAll("span[data-verify-url]").forEach((span) => {
      const url = span.dataset.verifyUrl;
      const verdict = linkCache.get(url);
      if (verdict === "ok") {
        const a = document.createElement("a");
        a.textContent = span.textContent;
        activateLink(a, url);
        span.replaceWith(a);
      } else if (verdict === "bad") {
        delete span.dataset.verifyUrl;
        span.className = "link-broken";
        span.title = brokenTitle(url);
      }
    });
  }

  function decorateCodeBlock(code, labelText) {
    const pre = code.parentElement;
    if (pre.querySelector(".code-head")) return;
    const lang = (code.className.match(/language-([\w+-]+)/) || [])[1] || "";
    const head = document.createElement("div");
    head.className = "code-head";
    const label = document.createElement("span");
    label.textContent = labelText || lang;
    const copy = document.createElement("button");
    copy.className = "code-copy";
    copy.textContent = "Copy";
    copy.addEventListener("click", () => {
      navigator.clipboard.writeText(code.textContent).then(() => {
        copy.textContent = "Copied";
        setTimeout(() => (copy.textContent = "Copy"), 1500);
      });
    });
    head.append(label, copy);
    pre.insertBefore(head, code);
  }

  // ── Scrolling ──────────────────────────────────────────────────
  // Stay pinned to the newest content unless the user deliberately scrolls
  // up; re-pin when they return to the bottom. Scroll events we trigger
  // ourselves are flagged so they can't be mistaken for the user.
  let autoScrolling = false;

  el.chatScroll.addEventListener("scroll", () => {
    const gap = el.chatScroll.scrollHeight - el.chatScroll.scrollTop - el.chatScroll.clientHeight;
    if (gap <= 40) {
      state.userScrolledUp = false;
    } else if (!autoScrolling) {
      state.userScrolledUp = true;
    }
  }, { passive: true });

  // Explicit upward gestures unpin immediately, even mid-stream
  el.chatScroll.addEventListener("wheel", (ev) => {
    if (ev.deltaY < 0) state.userScrolledUp = true;
  }, { passive: true });
  let touchStartY = 0;
  el.chatScroll.addEventListener("touchstart", (ev) => {
    touchStartY = ev.touches[0].clientY;
  }, { passive: true });
  el.chatScroll.addEventListener("touchmove", (ev) => {
    if (ev.touches[0].clientY > touchStartY + 10) state.userScrolledUp = true;
  }, { passive: true });

  function scrollToBottom(force) {
    if (force) state.userScrolledUp = false;
    if (state.userScrolledUp) return;
    autoScrolling = true;
    el.chatScroll.scrollTop = el.chatScroll.scrollHeight;
    requestAnimationFrame(() => { autoScrolling = false; });
  }

  // Keep the view pinned when heights change without a scroll: streamed
  // tokens, images loading, code-block decoration, the composer autosizing.
  const pinObserver = new ResizeObserver(() => scrollToBottom());
  pinObserver.observe(el.messages);
  pinObserver.observe(el.chatScroll);

  // ── Message DOM builders ───────────────────────────────────────
  function hideWelcome() { el.welcome.hidden = true; }
  function showWelcome() { el.welcome.hidden = false; }

  function addUserMessage(text) {
    hideWelcome();
    const msg = document.createElement("div");
    msg.className = "msg msg-user";
    const bubble = document.createElement("div");
    bubble.className = "msg-bubble";
    bubble.textContent = text;
    msg.appendChild(bubble);
    el.messages.appendChild(msg);
    scrollToBottom(true);
    return msg;
  }

  function addAssistantTurn() {
    hideWelcome();
    const msg = document.createElement("div");
    msg.className = "msg msg-assistant";
    const content = document.createElement("div");
    content.className = "msg-content";
    msg.appendChild(content);
    el.messages.appendChild(msg);
    return { root: msg, content };
  }

  function inlineUrl(url) {
    return url + (url.includes("?") ? "&" : "?") + "inline=1";
  }

  // A file of code the agent wrote: shown for inspection in its own scrollable
  // pane, with the same header, highlighting and Copy button as a code block in
  // the reply. The chip below it still downloads the real file.
  function addCodeInset(root, f) {
    const wrap = document.createElement("div");
    wrap.className = "code-file";
    const pre = document.createElement("pre");
    const code = document.createElement("code");
    code.className = "language-" + f.preview.language;
    code.textContent = f.preview.text;   // never innerHTML: this is file content
    pre.appendChild(code);
    wrap.appendChild(pre);

    if (f.preview.truncated) {
      const note = document.createElement("div");
      note.className = "code-truncated";
      note.textContent = "Preview truncated — download the file for the rest.";
      wrap.appendChild(note);
    }
    root.appendChild(wrap);

    try { hljs.highlightElement(code); } catch (e) { /* ignore */ }
    const lines = f.preview.text.replace(/\n$/, "").split("\n").length;
    decorateCodeBlock(code, `${f.name} · ${lines} line${lines === 1 ? "" : "s"}`);
  }

  // A page the agent generated, running in the reply. The frame is sandboxed
  // without allow-same-origin, so the page has an opaque origin: it cannot
  // touch this document, its cookies or its storage. The server pins that down
  // again with its own CSP (see _PREVIEW_CSP) — this attribute is the half of
  // the fence the browser enforces here.
  function addWebPreview(root, f) {
    const wrap = document.createElement("div");
    wrap.className = "web-preview";

    const head = document.createElement("div");
    head.className = "code-head";
    const label = document.createElement("span");
    label.textContent = f.name;
    const open = document.createElement("a");
    open.className = "code-copy";
    open.href = f.preview_url;
    open.target = "_blank";
    open.rel = "noopener noreferrer";
    open.textContent = "Open ↗";
    head.append(label, open);

    const box = document.createElement("div");
    box.className = "web-frame-box";      // drag the bottom edge to resize
    const frame = document.createElement("iframe");
    frame.className = "web-frame";
    frame.setAttribute("sandbox", "allow-scripts allow-modals allow-forms " +
                                  "allow-popups allow-pointer-lock allow-downloads");
    frame.setAttribute("referrerpolicy", "no-referrer");
    frame.title = f.name;
    frame.src = f.preview_url;
    box.appendChild(frame);

    // A page that fails inside the sandbox just looks blank from out here, so
    // the shim posts out what broke — a missing file, a first-line exception.
    const issues = document.createElement("div");
    issues.className = "web-issues";
    issues.hidden = true;
    const seen = new Set();
    window.addEventListener("message", (ev) => {
      if (ev.source !== frame.contentWindow) return;
      const d = ev.data;
      if (!d || d.__onit_preview !== "issue" || typeof d.text !== "string") return;
      if (seen.has(d.text) || seen.size >= 3) return;
      seen.add(d.text);
      const line = document.createElement("div");
      line.textContent = d.text;         // never innerHTML: this crossed a sandbox
      issues.appendChild(line);
      issues.hidden = false;
    });

    wrap.append(head, box, issues);
    root.appendChild(wrap);

    if (f.preview) {
      const details = document.createElement("details");
      details.className = "web-source";
      const summary = document.createElement("summary");
      summary.textContent = "Source";
      details.appendChild(summary);
      root.appendChild(details);
      addCodeInset(details, f);
    }
  }

  function addFileChips(root, files) {
    const all = (files || []).slice();
    if (!all.length) return;

    // An image already rendered in the reply (the model wrote its own
    // `![…](…)`) doesn't need a second copy below it — only its download chip.
    const shown = new Set(
      Array.from(root.querySelectorAll("img[src]"), (i) => i.getAttribute("src"))
    );
    // Same for code the model already quoted in full in its answer.
    const quoted = Array.from(root.querySelectorAll("pre > code"), (c) => c.textContent);
    const previews = all.filter(
      (f) => f.kind === "image" && !shown.has(inlineUrl(f.url))
    );

    if (previews.length) {
      const gallery = document.createElement("div");
      gallery.className = "file-images";
      for (const f of previews) {
        const a = document.createElement("a");
        a.className = "file-image";
        a.href = inlineUrl(f.url);       // full size in a new tab
        a.target = "_blank";
        a.rel = "noopener";
        const img = document.createElement("img");
        img.src = inlineUrl(f.url);
        img.alt = f.name;
        img.loading = "lazy";
        // A file the agent named but never wrote (or wrote as something other
        // than the image it claims) shouldn't leave a broken frame behind.
        img.addEventListener("error", () => a.remove());
        a.appendChild(img);
        gallery.appendChild(a);
      }
      root.appendChild(gallery);
    }

    for (const f of all) {
      if (f.kind === "web" && f.preview_url) {
        addWebPreview(root, f);
        continue;
      }
      if (f.kind !== "code" || !f.preview) continue;
      const body = f.preview.text.trim();
      if (body && quoted.some((q) => q.includes(body))) continue;
      addCodeInset(root, f);
    }

    const wrap = document.createElement("div");
    wrap.className = "file-chips";
    for (const f of all) {
      const a = document.createElement("a");
      a.className = "file-chip";
      a.href = f.url;
      a.setAttribute("download", f.name);
      const size = f.size ? ` <span class="file-size">${formatSize(f.size)}</span>` : "";
      const icon = f.kind === "image" ? "🖼️"
        : f.kind === "web" ? "🌐"
        : f.kind === "code" ? "📝" : "📄";
      a.innerHTML = `${icon} <span>${escapeHtml(f.name)}</span>${size}`;
      wrap.appendChild(a);
    }
    root.appendChild(wrap);
  }

  function addMeta(root, elapsed, tokS) {
    const parts = [];
    if (elapsed) parts.push(`${elapsed}s`);
    if (tokS) parts.push(`${tokS} tok/s`);
    if (!parts.length) return;
    const meta = document.createElement("div");
    meta.className = "msg-meta";
    meta.textContent = parts.join(" · ");
    root.appendChild(meta);
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  function formatSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / 1048576).toFixed(1) + " MB";
  }

  // ── API helpers ────────────────────────────────────────────────
  function apiHeaders(extra) {
    const h = Object.assign({}, extra);
    if (state.sessionId) h["X-Session-Id"] = state.sessionId;
    return h;
  }

  async function api(path, options) {
    const res = await fetch(path, Object.assign({ headers: apiHeaders() }, options, {
      headers: apiHeaders(options && options.headers),
    }));
    if (res.status === 401) { showLogin(); throw new Error("Not authenticated"); }
    return res;
  }

  // ── Analytics ──────────────────────────────────────────────────
  // Standard GA4 gtag snippet, injected only when the server config carries
  // a measurement ID (web_ga_measurement_id / ONIT_GA_MEASUREMENT_ID). The
  // SPA never rewrites the URL, so the initial page_view is the whole story.
  function initAnalytics(gaId) {
    if (!gaId || !/^G-[A-Z0-9]{4,16}$/.test(gaId)) return;
    const s = document.createElement("script");
    s.async = true;
    s.src = "https://www.googletagmanager.com/gtag/js?id=" + encodeURIComponent(gaId);
    document.head.appendChild(s);
    window.dataLayer = window.dataLayer || [];
    window.gtag = function () { window.dataLayer.push(arguments); };
    window.gtag("js", new Date());
    window.gtag("config", gaId);
  }

  // ── Boot / config ──────────────────────────────────────────────
  function showLogin() {
    el.app.hidden = true;
    el.login.hidden = false;
  }

  async function boot() {
    initTheme();
    let config;
    try {
      config = await (await fetch("/api/config")).json();
    } catch (e) {
      document.body.innerHTML = "<p style='padding:40px;font-family:sans-serif'>OnIt server unreachable.</p>";
      return;
    }
    state.config = config;
    initAnalytics(config.ga_id);
    document.title = config.title || "OnIt Chat";
    el.brandTitle.textContent = config.brand || "OnIt";
    el.composerHint.textContent = `${config.brand || "OnIt"} can make mistakes. Verify important results.`;
    el.loginTitle.textContent = config.title || "OnIt Chat";

    if (config.auth_enabled && !config.authenticated) {
      showLogin();
      return;
    }
    el.login.hidden = true;
    el.app.hidden = false;

    if (config.auth_enabled && config.email) {
      el.sidebarFoot.hidden = false;
      el.userEmail.textContent = config.email;
      el.userAvatar.textContent = config.email[0].toUpperCase();
    }
    if (config.show_logs) el.logsToggle.hidden = false;

    await loadHistory();
    await refreshSessions();
    el.input.focus();
  }

  // ── History / sessions ─────────────────────────────────────────
  async function loadHistory() {
    const res = await api("/api/history");
    const data = await res.json();
    state.sessionId = data.session_id;
    localStorage.setItem("onit.sid", state.sessionId);

    el.messages.innerHTML = "";
    if (!data.messages.length) showWelcome(); else hideWelcome();
    for (const m of data.messages) {
      if (m.role === "user") {
        addUserMessage(m.content);
      } else {
        const turn = addAssistantTurn();
        renderMarkdown(turn.content, m.content);
        addFileChips(turn.root, m.files);
      }
    }
    scrollToBottom(true);

    // A task may still be running for this session (e.g. after a page
    // refresh mid-generation): show the indicator and poll until done.
    if (data.processing && !state.processing) {
      setProcessing(true);
      const turn = addAssistantTurn();
      const chip = statusChip(turn.content, "Working…");
      pollWhileProcessing(turn, chip);
    }
  }

  function pollWhileProcessing(turn, chip) {
    clearInterval(state.pollTimer);
    state.pollTimer = setInterval(async () => {
      try {
        const data = await (await api("/api/history")).json();
        if (!data.processing) {
          clearInterval(state.pollTimer);
          setProcessing(false);
          await loadHistory();
          await refreshSessions();
        }
      } catch (e) { /* keep polling */ }
    }, 2000);
  }

  async function refreshSessions() {
    let data;
    try {
      data = await (await api("/api/sessions")).json();
    } catch (e) { return; }
    el.sessionList.innerHTML = "";
    for (const s of data.sessions) {
      const item = document.createElement("div");
      item.className = "session-item" + (s.session_id === state.sessionId ? " active" : "");
      const title = document.createElement("span");
      title.className = "session-title";
      title.textContent = s.tag || s.preview || "New chat";
      item.appendChild(title);
      if (s.processing) {
        const dot = document.createElement("span");
        dot.className = "session-busy";
        item.appendChild(dot);
      }
      const actions = document.createElement("span");
      actions.className = "session-actions";
      actions.append(
        sessionAction("✎", "Rename", () => renameSession(s)),
        sessionAction("✕", "Delete", () => deleteSession(s)),
      );
      item.appendChild(actions);
      item.addEventListener("click", (ev) => {
        if (ev.target.closest(".session-actions")) return;
        switchSession(s.session_id);
      });
      el.sessionList.appendChild(item);
    }
  }

  function sessionAction(glyph, label, onClick) {
    const btn = document.createElement("button");
    btn.className = "session-action-btn";
    btn.title = label;
    btn.textContent = glyph;
    btn.addEventListener("click", (ev) => { ev.stopPropagation(); onClick(); });
    return btn;
  }

  async function switchSession(sid) {
    if (state.processing) return;
    state.sessionId = sid;
    localStorage.setItem("onit.sid", sid);
    forgetEmailVerdicts();
    clearAttachments();
    await loadHistory();
    await refreshSessions();
    el.input.focus();
  }

  async function newSession() {
    if (state.processing) return;
    const res = await api("/api/sessions/new", { method: "POST" });
    const data = await res.json();
    await switchSession(data.session_id);
  }

  async function clearChat() {
    if (state.processing) return;
    if (!confirm("Clear this chat's history?")) return;
    await api("/api/clear", { method: "POST" });
    forgetEmailVerdicts();
    clearAttachments();
    await loadHistory();
    await refreshSessions();
    el.input.focus();
  }

  async function clearAllSessions() {
    if (state.processing) return;
    if (!confirm("Delete all chats? This cannot be undone.")) return;
    const res = await api("/api/sessions", { method: "DELETE" });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      alert(err.detail || "Failed to delete chats");
      return;
    }
    localStorage.removeItem("onit.sid");
    state.sessionId = null;
    clearAttachments();
    await loadHistory();  // creates a fresh session
    await refreshSessions();
    el.input.focus();
  }

  async function renameSession(s) {
    const tag = prompt("Rename chat:", s.tag || "");
    if (!tag) return;
    const res = await api(`/api/sessions/${s.session_id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tag }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      alert(err.detail || "Rename failed");
    }
    await refreshSessions();
  }

  async function deleteSession(s) {
    if (!confirm(`Delete chat "${s.tag || s.preview || s.session_id.slice(0, 8)}"?`)) return;
    await api(`/api/sessions/${s.session_id}`, { method: "DELETE" });
    if (s.session_id === state.sessionId) {
      localStorage.removeItem("onit.sid");
      state.sessionId = null;
      await loadHistory();  // creates a fresh session
    }
    await refreshSessions();
  }

  // ── Status chip ────────────────────────────────────────────────
  function statusChip(parent, text) {
    const chip = document.createElement("div");
    chip.className = "status-chip";
    chip.innerHTML = `<span class="status-dot"></span><span class="status-text"></span>`;
    chip.querySelector(".status-text").textContent = text;
    parent.appendChild(chip);
    return {
      set(t) {
        chip.querySelector(".status-text").textContent = t;
        chip.hidden = !t;
      },
      remove() { chip.remove(); },
    };
  }

  // ── Sending / SSE ──────────────────────────────────────────────
  function setProcessing(on) {
    state.processing = on;
    el.stopBtn.hidden = !on;
    el.sendBtn.hidden = on;
    updateSendEnabled();
  }

  function updateSendEnabled() {
    el.sendBtn.disabled = state.processing ||
      (!el.input.value.trim() && !state.attachments.length);
  }

  async function send() {
    const text = el.input.value.trim();
    if ((!text && !state.attachments.length) || state.processing) return;

    const files = state.attachments.map((a) => a.name);
    let display = text;
    if (files.length) display += (display ? "\n" : "") + files.map((f) => `📎 ${f}`).join("\n");
    addUserMessage(display);

    el.input.value = "";
    autosize();
    clearAttachments();
    setProcessing(true);

    const turn = addAssistantTurn();
    const chip = statusChip(turn.root, "Thinking…");
    let streamBlock = null;   // element receiving live tokens
    let streamText = "";
    let rafPending = false;

    const paintStream = () => {
      if (rafPending || !streamBlock) return;
      rafPending = true;
      requestAnimationFrame(() => {
        rafPending = false;
        if (!streamBlock) return;
        renderMarkdown(streamBlock, streamText);
        const cursor = document.createElement("span");
        cursor.className = "stream-cursor";
        (streamBlock.lastElementChild || streamBlock).appendChild(cursor);
        scrollToBottom();
      });
    };

    const ensureStreamBlock = () => {
      if (!streamBlock) {
        streamBlock = document.createElement("div");
        turn.content.appendChild(streamBlock);
      }
    };

    const handlers = {
      token(d) {
        chip.set("");
        ensureStreamBlock();
        streamText += d.delta || "";
        paintStream();
      },
      phase_end(d) {
        // Tools ran since the last render, so the session may have picked up
        // sources that ground an address previously judged ungrounded.
        forgetEmailVerdicts();
        // Commit the streamed phase and prepare for the next one
        if (streamBlock) {
          renderMarkdown(streamBlock, d.content || streamText);
          streamBlock = null;
          streamText = "";
        }
        chip.set("Working…");
      },
      status(d) {
        chip.set(d.text || "");
        if (!d.text && state.processing) chip.set("Working…");
      },
      answer_start() {
        // Everything committed so far was the model working, not answering.
        // Clear it so the answer starts on a clean turn: `done` would replace
        // it anyway, and until then the preamble reads as if it were the reply.
        turn.content.innerHTML = "";
        streamBlock = null;
        streamText = "";
        chip.set("Writing the answer…");
      },
      done(d) {
        chip.remove();
        forgetEmailVerdicts();
        // Final response supersedes streamed phases — unless it is empty, in
        // which case wiping would leave the turn blank. Keep what streamed.
        if ((d.content || "").trim()) {
          turn.content.innerHTML = "";
          const final = document.createElement("div");
          turn.content.appendChild(final);
          renderMarkdown(final, d.content);
        } else if (streamBlock) {
          renderMarkdown(streamBlock, streamText);
        }
        addFileChips(turn.root, d.files);
        addMeta(turn.root, d.elapsed, d.tok_s);
        streamBlock = null;
      },
      error(d) {
        chip.remove();
        const err = document.createElement("div");
        err.className = "msg-error";
        err.textContent = d.message || "Something went wrong.";
        turn.content.appendChild(err);
        streamBlock = null;
      },
    };

    try {
      const res = await api("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, files }),
      });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        handlers.error({ message: errBody.detail || `Request failed (${res.status})` });
        return;
      }
      await readSSE(res, (event, data) => {
        (handlers[event] || (() => {}))(data);
        scrollToBottom();
      });
    } catch (e) {
      // Connection dropped mid-stream: fall back to history polling
      chip.set("Reconnecting…");
      pollWhileProcessing(turn, chip);
      return;
    } finally {
      if (!state.pollTimer || !state.processing) setProcessing(false);
    }
    setProcessing(false);
    refreshSessions();
    el.input.focus();
  }

  async function readSSE(res, onEvent) {
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const raw = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        let event = "message";
        const dataLines = [];
        for (const line of raw.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
          // lines starting with ":" are keepalive comments
        }
        if (!dataLines.length) continue;
        let data = {};
        try { data = JSON.parse(dataLines.join("\n")); } catch (e) { continue; }
        onEvent(event, data);
      }
    }
  }

  async function stop() {
    try { await api("/api/chat/stop", { method: "POST" }); } catch (e) { /* ignore */ }
  }

  // ── Attachments ────────────────────────────────────────────────
  function renderAttachments() {
    el.attachments.innerHTML = "";
    el.attachments.hidden = !state.attachments.length;
    state.attachments.forEach((a, i) => {
      const chip = document.createElement("span");
      chip.className = "attachment-chip";
      chip.innerHTML = `📎 <span>${escapeHtml(a.name)}</span>`;
      const rm = document.createElement("button");
      rm.className = "attachment-remove";
      rm.textContent = "✕";
      rm.title = "Remove";
      rm.addEventListener("click", () => {
        state.attachments.splice(i, 1);
        renderAttachments();
        updateSendEnabled();
      });
      chip.appendChild(rm);
      el.attachments.appendChild(chip);
    });
  }

  function clearAttachments() {
    state.attachments = [];
    renderAttachments();
  }

  async function uploadFile(file) {
    const form = new FormData();
    form.append("file", file);
    const res = await api("/api/upload", { method: "POST", body: form });
    if (!res.ok) { alert("Upload failed"); return; }
    const data = await res.json();
    state.sessionId = data.session_id;
    localStorage.setItem("onit.sid", state.sessionId);
    state.attachments.push({ name: data.name, url: data.url });
    renderAttachments();
    updateSendEnabled();
  }

  // ── Logs drawer ────────────────────────────────────────────────
  async function refreshLogs() {
    try {
      const data = await (await api("/api/logs")).json();
      el.logsBody.innerHTML = "";
      for (const log of data.logs) {
        const line = document.createElement("div");
        line.className = `log-line log-${log.level}`;
        line.textContent = `[${log.timestamp}] ${log.message}`;
        el.logsBody.appendChild(line);
      }
      el.logsBody.scrollTop = el.logsBody.scrollHeight;
    } catch (e) { /* ignore */ }
  }

  function toggleLogs(open) {
    state.logsOpen = open === undefined ? !state.logsOpen : open;
    el.logsDrawer.hidden = !state.logsOpen;
    clearInterval(state.logsTimer);
    if (state.logsOpen) {
      refreshLogs();
      state.logsTimer = setInterval(refreshLogs, 2000);
    }
  }

  // ── Theme ──────────────────────────────────────────────────────
  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    $("hljs-light").media = theme === "dark" ? "not all" : "all";
    $("hljs-dark").media = theme === "dark" ? "all" : "not all";
    localStorage.setItem("onit.theme", theme);
  }

  function initTheme() {
    const saved = localStorage.getItem("onit.theme");
    const preferred = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    applyTheme(saved || preferred);
  }

  // ── Composer behavior ──────────────────────────────────────────
  function autosize() {
    el.input.style.height = "auto";
    el.input.style.height = Math.min(el.input.scrollHeight, 220) + "px";
  }

  el.input.addEventListener("input", () => { autosize(); updateSendEnabled(); });
  el.input.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey) {
      ev.preventDefault();
      send();
    }
  });
  el.sendBtn.addEventListener("click", send);
  el.stopBtn.addEventListener("click", stop);
  el.attachBtn.addEventListener("click", () => el.fileInput.click());
  el.fileInput.addEventListener("change", () => {
    if (el.fileInput.files.length) uploadFile(el.fileInput.files[0]);
    el.fileInput.value = "";
  });
  el.newChat.addEventListener("click", newSession);
  el.clearChat.addEventListener("click", clearChat);
  el.clearAll.addEventListener("click", clearAllSessions);
  el.themeToggle.addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme");
    applyTheme(cur === "dark" ? "light" : "dark");
  });
  el.logsToggle.addEventListener("click", () => toggleLogs());
  el.logsClose.addEventListener("click", () => toggleLogs(false));
  el.sidebarClose.addEventListener("click", () => {
    el.sidebar.classList.add("collapsed");
    el.sidebarOpen.hidden = false;
  });
  el.sidebarOpen.addEventListener("click", () => {
    el.sidebar.classList.remove("collapsed");
    el.sidebarOpen.hidden = true;
  });

  // drag & drop upload
  document.addEventListener("dragover", (ev) => ev.preventDefault());
  document.addEventListener("drop", (ev) => {
    ev.preventDefault();
    if (ev.dataTransfer.files.length) uploadFile(ev.dataTransfer.files[0]);
  });

  boot();
})();

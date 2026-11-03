/* Edgent local UI: reader + tutor + dashboards. Plain JS, no external scripts (CSP). */
(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const token = new URLSearchParams(location.search).get("token");
  const api = async (path, opts = {}) => {
    const headers = { "Content-Type": "application/json" };
    if (token) headers["X-Edgent-Token"] = token;
    const r = await fetch(path, { ...opts, headers });
    const data = await r.json().catch(() => ({ error: { message: "bad json" } }));
    if (!r.ok) throw new Error((data.error && data.error.message) || r.statusText);
    return data;
  };
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  const state = { session: null, chapter: null, screen: null, lastEvent: 0, busy: false };

  // ------------------------------------------------------------ tabs
  $$("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  function showTab(name) {
    $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
    $$(".tab").forEach((t) => t.classList.toggle("active", t.id === "tab-" + name));
    ({ outline: loadOutline, build: loadBuild, learner: loadLearner, unknowns: loadUnknowns, sources: loadSources, doctor: loadDoctor }[name] || (() => {}))();
  }

  // ------------------------------------------------------------ workspace, chapters, sessions
  async function loadWorkspace() {
    const ws = await api("/api/workspace");
    $("#ws-name").textContent = ws.name || "";
  }
  async function loadChapters() {
    const chapters = await api("/api/chapters");
    const ul = $("#chapter-list");
    ul.innerHTML = "";
    chapters.forEach((c) => {
      const li = document.createElement("li");
      li.textContent = c.title || c.path;
      li.title = c.path;
      li.addEventListener("click", () => openChapter(c.path, li));
      ul.appendChild(li);
    });
    if (!chapters.length) ul.innerHTML = '<li class="muted">no chapters yet</li>';
  }
  async function openChapter(path, li) {
    const data = await api("/api/chapter?path=" + encodeURIComponent(path));
    state.chapter = path;
    $$("#chapter-list li").forEach((x) => x.classList.toggle("active", x === li));
    const art = $("#chapter");
    art.innerHTML = data.html;
    showTab("study");
    $$(".cite", art).forEach((el) => el.addEventListener("click", () => showLocator(el.dataset.locator)));
    updateScreen();
  }
  async function showLocator(loc) {
    const d = await api("/api/locator?locator=" + encodeURIComponent(loc));
    $("#locator-title").textContent = loc + (d.found ? "" : " (not found)");
    $("#locator-text").textContent = d.text || d.coverage_note || "";
    $("#locator-panel").classList.remove("hidden");
  }
  $("#locator-close").addEventListener("click", () => $("#locator-panel").classList.add("hidden"));

  async function loadSessions() {
    const sessions = await api("/api/sessions");
    const ul = $("#session-list");
    ul.innerHTML = "";
    sessions.filter((s) => s.mode !== "subagent").forEach((s) => {
      const li = document.createElement("li");
      li.textContent = s.title || s.session_id;
      li.classList.toggle("active", s.session_id === state.session);
      li.addEventListener("click", () => selectSession(s.session_id));
      ul.appendChild(li);
    });
    if (!state.session && sessions.length) selectSession(sessions.find((s) => s.mode !== "subagent")?.session_id);
  }
  $("#new-session").addEventListener("click", async () => {
    const s = await api("/api/sessions", { method: "POST", body: JSON.stringify({ mode: "chat", title: "Study session" }) });
    await loadSessions();
    selectSession(s.session_id);
  });
  async function selectSession(id) {
    if (!id) return;
    state.session = id;
    $$("#session-list li").forEach((x) => x.classList.toggle("active", x.textContent === id || x.title === id));
    const msgs = await api(`/api/sessions/${encodeURIComponent(id)}/messages?limit=200`);
    renderMessages(msgs);
  }
  function renderMessages(msgs) {
    const box = $("#messages");
    box.innerHTML = "";
    msgs.forEach((m) => appendMessage(m));
    box.scrollTop = box.scrollHeight;
  }
  function appendMessage(m) {
    const div = document.createElement("div");
    div.className = "msg " + m.role;
    const text = m.text || (m.tool_calls ? m.tool_calls.map((t) => `→ ${t.name}(${JSON.stringify(t.arguments).slice(0, 120)})`).join("\n") : "");
    div.textContent = m.role === "tool" ? `[${m.name}] ${String(text).slice(0, 300)}` : text;
    $("#messages").appendChild(div);
    $("#messages").scrollTop = $("#messages").scrollHeight;
    return div;
  }

  // ------------------------------------------------------------ what is on screen
  function updateScreen() {
    const sel = String(window.getSelection && window.getSelection()).trim().slice(0, 1200);
    const reader = $("#reader");
    let visible = null;
    $$("section[data-section]", $("#chapter")).forEach((s) => {
      const r = s.getBoundingClientRect();
      if (!visible && r.bottom > 0 && r.top < reader.clientHeight) visible = s.dataset.section;
    });
    state.screen = state.chapter ? { chapter: state.chapter, section: visible, selection: sel || null } : null;
    $("#screen-hint").textContent = state.screen ? `on screen: ${state.chapter}${visible ? " › " + visible : ""}${sel ? " · selection: “" + sel.slice(0, 60) + (sel.length > 60 ? "…" : "") + "”" : ""}` : "";
  }
  document.addEventListener("mouseup", updateScreen);
  $("#reader").addEventListener("scroll", () => { clearTimeout(state._t); state._t = setTimeout(updateScreen, 150); });

  // ------------------------------------------------------------ chat
  $("#chat-input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#chat-form").requestSubmit(); } });
  $("#chat-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = $("#chat-input").value.trim();
    if (!text || state.busy) return;
    if (!state.session) { const s = await api("/api/sessions", { method: "POST", body: JSON.stringify({ mode: "chat", title: "Study session" }) }); state.session = s.session_id; await loadSessions(); }
    state.busy = true; $("#send").disabled = true;
    appendMessage({ role: "user", text });
    $("#chat-input").value = "";
    const pending = appendMessage({ role: "assistant", text: "…" });
    try {
      const r = await api(`/api/sessions/${encodeURIComponent(state.session)}/send`, { method: "POST", body: JSON.stringify({ text, screen: state.screen }) });
      pending.textContent = r.text || `(${r.stop_reason})`;
      if (r.stop_reason && r.stop_reason !== "done") pending.textContent += `\n[stopped: ${r.stop_reason}]`;
    } catch (err) { pending.textContent = "error: " + err.message; }
    state.busy = false; $("#send").disabled = false;
  });

  // ------------------------------------------------------------ dashboards
  async function loadOutline() {
    const o = await api("/api/outline");
    const units = o.units || [];
    $("#tab-outline").innerHTML = `<h2>${esc(o.title || "Outline")}</h2><p class="muted">${esc(o.persona || "")}</p>` +
      (units.length ? "<ul>" + units.map((u) => `<li><span class="badge ${esc(u.status)}">${esc(u.status)}</span> <code>${esc(u.id)}</code> ${esc(u.kind)}: ${esc(u.title)}${u.open_obligations && u.open_obligations.length ? ` <span class="muted">(${u.open_obligations.length} open)</span>` : ""}</li>`).join("") + "</ul>" : '<p class="muted">No units yet. Start a build.</p>');
  }
  async function loadBuild() { const b = await api("/api/build"); $("#build-status").textContent = JSON.stringify(b, null, 2); }
  $("#build-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const goal = $("#build-goal").value.trim();
    if (!goal) return;
    try { const r = await api("/api/build/start", { method: "POST", body: JSON.stringify({ goal }) }); $("#build-status").textContent = JSON.stringify(r, null, 2); }
    catch (err) { $("#build-status").textContent = "error: " + err.message; }
  });
  async function loadLearner() {
    const l = await api("/api/learner");
    const p = l.progress || {};
    const due = l.due || [];
    const concepts = Object.entries(l.mastery || {}).sort((a, b) => (b[1].level || 0) - (a[1].level || 0));
    $("#tab-learner").innerHTML = `<h2>Learner</h2>
      <div><span class="stat"><b>${p.streak_days ?? 0}</b>day streak</span><span class="stat"><b>${p.points ?? 0}</b>points</span><span class="stat"><b>${due.length}</b>due reviews</span></div>
      <p>${(p.badges || []).map((b) => `<span class="badge">${esc(b.name || b.id)}</span>`).join(" ") || '<span class="muted">no badges yet</span>'}</p>
      <h3>Due now</h3>${due.length ? "<ul>" + due.map((d) => `<li>${esc(d.concept || d)}</li>`).join("") + "</ul>" : '<p class="muted">nothing due</p>'}
      <h3>Mastery</h3>${concepts.length ? "<table><tr><th>concept</th><th>level</th><th>next review</th></tr>" + concepts.map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc((v.level ?? 0).toFixed ? v.level.toFixed(1) : v.level)}</td><td>${esc(v.next_review || "")}</td></tr>`).join("") + "</table>" : '<p class="muted">no concepts tracked yet</p>'}`;
  }
  async function loadUnknowns() {
    const u = await api("/api/unknowns");
    $("#tab-unknowns").innerHTML = `<h2>Unknowns</h2><pre class="box">${esc(u.markdown || "")}</pre>`;
  }
  async function loadSources() {
    const s = await api("/api/sources");
    $("#tab-sources").innerHTML = "<h2>Sources</h2>" + (s.length ? "<table><tr><th>id</th><th>kind</th><th>title</th><th>coverage</th></tr>" + s.map((x) => `<tr><td><code>${esc(x.id)}</code></td><td>${esc(x.kind)}</td><td>${esc(x.title)}</td><td>${esc((x.coverage || {}).status || "")}</td></tr>`).join("") + "</table>" : '<p class="muted">No sources. Add one with <code>edgent sources add &lt;file or url&gt;</code>.</p>');
  }
  async function loadDoctor() {
    const d = await api("/api/doctor");
    const items = d.items || [];
    $("#tab-doctor").innerHTML = "<h2>Doctor</h2><table><tr><th>tool</th><th>status</th><th>hint</th></tr>" + items.map((i) => `<tr><td>${esc(i.name)}</td><td>${i.available ? "ok" : "missing"}</td><td class="muted">${esc(i.hint || i.path || "")}</td></tr>`).join("") + "</table>" + `<h3>Provider</h3><pre class="box">${esc(JSON.stringify(d.provider || {}, null, 2))}</pre>`;
  }

  // ------------------------------------------------------------ progress events (polling)
  async function poll() {
    try {
      const evs = await api("/api/events?since=" + state.lastEvent);
      evs.forEach((e) => { state.lastEvent = Math.max(state.lastEvent, e.seq || 0); if (e.kind === "tool_call" && state.busy) appendMessage({ role: "tool", name: e.name, text: e.detail || "" }); });
    } catch (e) { /* ignore */ }
    setTimeout(poll, state.busy ? 1500 : 6000);
  }

  loadWorkspace().then(loadChapters).then(loadSessions).then(poll).catch((e) => { $("#chapter").innerHTML = `<p class="muted">${esc(e.message)}</p>`; });
})();

// The page: dashboard (from /api/dashboard), upload, and a live view of one agent run (/ws/run).
// All numbers come from the server (tools.py) - the page only shows them.

const $ = (id) => document.getElementById(id);

const TOOLS = {
  list_documents: { icon: "📂", label: "בודק מה מחכה בתיבה" },
  read_document: { icon: "🔍", label: "קורא את המסמך" },
  find_duplicate: { icon: "🧩", label: "בודק שהמסמך לא נרשם כבר" },
  add_transaction: { icon: "✍️", label: "רושם את התנועה" },
  skip_document: { icon: "⏭️", label: "מדלג על המסמך" },
  get_monthly_summary: { icon: "📊", label: "מסכם את החודש" },
};
// One color per expense category, by its place in config.EXPENSE_CATEGORIES (stable between months)
const PALETTE = ["#6366f1", "#14b8a6", "#f59e0b", "#ec4899", "#0ea5e9", "#8b5cf6",
                 "#10b981", "#f97316", "#e11d48", "#84cc16", "#a16207", "#94a3b8"];

const state = {
  month: `${new Date().getFullYear()}-${String(new Date().getMonth() + 1).padStart(2, "0")}`,  // local time, not UTC
  data: null,
  file: null,
  running: false,
  newIds: new Set(),   // transactions recorded in this run - highlighted in the table
  firstLoad: true,     // an empty current month on opening -> show the latest month with data instead
  charts: {},
};

const moneyFormat = new Intl.NumberFormat("he-IL", { style: "currency", currency: "ILS", maximumFractionDigits: 2 });
const money = (n) => moneyFormat.format(n);
const monthName = (m) => new Intl.DateTimeFormat("he-IL", { month: "long", year: "numeric" })
  .format(new Date(Number(m.slice(0, 4)), Number(m.slice(5, 7)) - 1, 1));
const shortMonth = (m) => new Intl.DateTimeFormat("he-IL", { month: "short" })
  .format(new Date(Number(m.slice(0, 4)), Number(m.slice(5, 7)) - 1, 1));
const formatDate = (d) => d.split("-").reverse().join(".");
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// A Latin file name inside a Hebrew line came out scrambled ("jpeg.003721_20261026"): wrap it in
// Unicode isolate marks (LRI ... PDI) so it is laid out left-to-right as one unit.
const FILE_NAME = /[\w\-]+(?:[ .()\-][\w\-()]+)*\.(?:jpe?g|png|webp)\b/gi;
const isolateFileNames = (text) => String(text).replace(FILE_NAME, "\u2066$&\u2069");

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = isolateFileNames(text);
  return node;
}

function showPageError(text) {
  $("pageError").textContent = text;
  $("pageError").hidden = !text;
}

// ---------- Dashboard ----------

async function loadDashboard() {
  try {
    const response = await fetch(`/api/dashboard?month=${state.month}`);
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || body.detail || "שגיאה בטעינת הנתונים");
    if (state.firstLoad && !body.transactions.length && body.latest_month && body.latest_month !== state.month) {
      state.firstLoad = false;
      state.month = body.latest_month;
      return loadDashboard();
    }
    state.firstLoad = false;
    state.data = body;
    await document.fonts.ready;  // charts measure their labels: with a fallback font they were cut off
    showPageError("");
    renderDashboard();
  } catch (e) {
    showPageError(e.message === "Failed to fetch" ? "אין חיבור לשרת. האם web_app.py רץ?" : e.message);
  }
}

function renderDashboard() {
  const { summary, transactions } = state.data;
  renderMonths();
  countUp($("kpiIncome"), summary.income_total);
  countUp($("kpiExpense"), summary.expense_total);
  countUp($("kpiBalance"), summary.balance);
  $("kpiCount").textContent = summary.transaction_count ? `${summary.transaction_count} תנועות בחודש` : "אין תנועות בחודש הזה";
  renderDonut(summary);
  renderBudgets(summary.budgets);
  renderTrend(state.data.trend);
  renderTable(transactions);
  renderPendingHint();
}

function countUp(node, target) {
  const start = Number(node.dataset.value || 0);
  node.dataset.value = target;
  const began = performance.now();
  const frame = (now) => {
    const t = Math.min(1, (now - began) / 700);
    const eased = 1 - Math.pow(1 - t, 3);
    node.textContent = money(start + (target - start) * eased);
    if (t < 1) requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

function renderMonths() {
  const select = $("monthSelect");
  const months = new Set(state.data.months);
  months.add(state.month);
  select.replaceChildren(...[...months].sort().reverse().map((m) => {
    const option = el("option", "", monthName(m));
    option.value = m;
    return option;
  }));
  select.value = state.month;
}

function shiftMonth(delta) {
  const [y, m] = state.month.split("-").map(Number);
  const d = new Date(y, m - 1 + delta, 1);
  state.month = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`;
  loadDashboard();
}

function categoryColor(category) {
  const index = state.data.categories.indexOf(category);
  return PALETTE[(index >= 0 ? index : PALETTE.length - 1) % PALETTE.length];
}

function renderDonut(summary) {
  const entries = Object.entries(summary.expenses_by_category)
    .filter(([, amount]) => amount > 0)
    .sort((a, b) => b[1] - a[1]);
  $("donutTotal").textContent = money(summary.expense_total);

  const legend = $("legend");
  legend.replaceChildren();
  if (!entries.length) legend.append(el("li", "empty", "אין עדיין הוצאות בחודש הזה"));
  for (const [category, amount] of entries) {
    const item = el("li");
    const dot = el("span", "dot");
    dot.style.background = categoryColor(category);
    const share = summary.expense_total > 0 ? ` · ${Math.round((amount / summary.expense_total) * 100)}%` : "";
    item.append(dot, el("span", "", category + share), el("span", "amount", money(amount)));
    legend.append(item);
  }

  const data = entries.length ? entries.map(([, a]) => a) : [1];
  const colors = entries.length ? entries.map(([c]) => categoryColor(c)) : [cssVar("--border")];
  const labels = entries.length ? entries.map(([c]) => c) : ["אין הוצאות"];
  if (state.charts.donut) state.charts.donut.destroy();
  state.charts.donut = new Chart($("donutChart"), {
    type: "doughnut",
    data: { labels, datasets: [{ data, backgroundColor: colors, borderColor: cssVar("--surface"), borderWidth: 3, hoverOffset: 6 }] },
    options: {
      cutout: "72%",
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { enabled: entries.length > 0, rtl: true, callbacks: { label: (c) => ` ${c.label}: ${money(c.raw)}` } },
      },
    },
  });
}

function renderBudgets(budgets) {
  const list = $("budgets");
  list.replaceChildren();
  for (const b of budgets) {
    const item = el("li", "budget");
    item.dataset.status = b.status;
    const top = el("div", "budget-top");
    const name = el("span", "", b.category);
    if (b.status === "near") name.append(el("span", "tag near", "קרוב לתקציב"));
    if (b.status === "over") name.append(el("span", "tag over", "חריגה!"));
    top.append(name, el("span", "nums", `${money(b.spent)} / ${money(b.budget)} · ${b.percent}%`));
    const bar = el("div", "bar");
    const fill = el("div", "bar-fill");
    bar.append(fill);
    item.append(top, bar);
    list.append(item);
    requestAnimationFrame(() => { fill.style.width = `${Math.min(100, Math.max(0, b.percent))}%`; });
  }
}

function renderTrend(trend) {
  if (state.charts.trend) state.charts.trend.destroy();
  const grid = cssVar("--border");
  const text = cssVar("--muted");
  state.charts.trend = new Chart($("trendChart"), {
    type: "bar",
    data: {
      labels: trend.map((t) => shortMonth(t.month)),
      datasets: [
        { label: "הכנסות", data: trend.map((t) => t.income), backgroundColor: cssVar("--income"), borderRadius: 6, maxBarThickness: 22 },
        { label: "הוצאות", data: trend.map((t) => t.expense), backgroundColor: cssVar("--expense"), borderRadius: 6, maxBarThickness: 22 },
      ],
    },
    options: {
      maintainAspectRatio: false,
      scales: {
        x: { reverse: true, grid: { display: false }, ticks: { color: text, font: { family: "Rubik" } } },
        y: { position: "right", grid: { color: grid }, border: { display: false },
             ticks: { color: text, font: { family: "Rubik" }, callback: (v) => `₪${Number(v).toLocaleString("he-IL")}` } },
      },
      plugins: {
        legend: { rtl: true, labels: { color: text, usePointStyle: true, pointStyle: "rectRounded", font: { family: "Rubik" } } },
        tooltip: { rtl: true, callbacks: { label: (c) => ` ${c.dataset.label}: ${money(c.raw)}` } },
      },
    },
  });
}

function renderTable(transactions) {
  const body = $("txBody");
  body.replaceChildren();
  $("txCount").textContent = transactions.length ? `${transactions.length} תנועות` : "";
  if (!transactions.length) {
    const row = el("tr");
    const cell = el("td", "empty", "אין תנועות בחודש הזה. העלי מסמך כדי להתחיל.");
    cell.colSpan = 5;
    row.append(cell);
    body.append(row);
    return;
  }
  for (const t of transactions) {
    const row = el("tr", state.newIds.has(t.id) ? "new" : "");
    const business = el("td", "biz", t.business);
    if (t.items && t.items.length) business.append(el("span", "items", t.items.slice(0, 4).join(" · ")));
    const chip = el("span", "chip");
    const dot = el("span", "dot");
    dot.style.background = t.type === "income" ? cssVar("--income") : categoryColor(t.category);
    chip.append(dot, document.createTextNode(t.category));
    const category = el("td");
    category.append(chip);
    // A refund is an expense with a negative amount: money that came back
    const back = t.type === "income" || t.amount < 0;
    const amount = el("td", `num ${back ? "income" : ""}`, `${back ? "+" : "−"}${money(Math.abs(t.amount))}`);
    if (t.amount < 0) amount.prepend(el("span", "tag refund", "זיכוי"));
    const view = el("td", "row-actions");
    if (t.has_image) {
      const button = el("button", "view-btn", "🧾");
      button.title = "הצגת המסמך המקורי";
      button.addEventListener("click", () => openImage(t.source_file));
      view.append(button);
    }
    const edit = el("button", "view-btn", "✏️");
    edit.title = "תיקון שם העסק או הקטגוריה";
    edit.disabled = state.running;
    edit.addEventListener("click", () => editRow(t, business, category, view));
    view.append(edit);
    row.append(el("td", "date", formatDate(t.date)), business, category, amount, view);
    body.append(row);
  }
}

// The image reading can be wrong (a test read "שרותי מזון" as "שירותי מיזוג"); her correction is
// saved and also replaces what the agent remembers about this business.
function editRow(t, businessCell, categoryCell, actionsCell) {
  const input = el("input", "edit-input");
  input.value = t.business;
  const select = el("select", "edit-input");
  const categories = t.type === "income" ? state.data.income_categories : state.data.categories;
  for (const c of categories) {
    const option = el("option", "", c);
    option.value = c;
    select.append(option);
  }
  select.value = t.category;
  businessCell.replaceChildren(input);
  categoryCell.replaceChildren(select);

  const save = el("button", "btn btn-primary btn-small", "שמירה");
  const cancel = el("button", "btn btn-ghost btn-small", "ביטול");
  cancel.addEventListener("click", () => renderTable(state.data.transactions));
  save.addEventListener("click", async () => {
    save.disabled = true;
    try {
      const response = await fetch(`/api/transactions/${encodeURIComponent(t.id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ business: input.value, category: select.value }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || "התיקון נכשל");
      toast("✓ תוקן. האג'נט יזכור את זה בפעם הבאה");
      loadDashboard();
    } catch (e) {
      toast(e.message, true);
      save.disabled = false;
    }
  });
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") save.click(); if (e.key === "Escape") cancel.click(); });
  actionsCell.replaceChildren(save, cancel);
  input.focus();
  input.select();
}

function toast(text, isError = false) {
  const note = el("div", `toast ${isError ? "toast-error" : ""}`, text);
  document.body.append(note);
  setTimeout(() => note.classList.add("toast-hide"), 3000);
  setTimeout(() => note.remove(), 3500);
}

function renderPendingHint() {
  const pending = state.data.inbox_pending;
  const hint = $("pendingHint");
  hint.hidden = state.running || !pending;
  if (!pending) return;
  hint.replaceChildren(
    document.createTextNode(`${pending === 1 ? "מסמך אחד מחכה" : `${pending} מסמכים מחכים`} בתיבה מהעלאה קודמת – האג'נט יטפל גם בהם. `),
  );
  const run = el("button", "link", "לטפל בהם עכשיו");
  run.addEventListener("click", () => startRun());
  hint.append(run);
}

function openImage(file) {
  $("modalImg").src = `/api/files/${encodeURIComponent(file)}`;
  $("imageModal").showModal();
}

// ---------- Upload ----------

const SUPPORTED = ["jpg", "jpeg", "png", "webp"];

function chooseFile(file) {
  if (!file || state.running) return;
  const extension = file.name.split(".").pop().toLowerCase();
  if (!SUPPORTED.includes(extension)) {
    showPageError(`אפשר להעלות רק תמונות (${SUPPORTED.join(", ")}). קובץ PDF אפשר לצלם מסך ולהעלות את הצילום.`);
    return;
  }
  showPageError("");
  state.file = file;
  $("previewImg").src = URL.createObjectURL(file);
  $("previewName").textContent = file.name;
  $("previewSize").textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB`;
  $("dzEmpty").hidden = true;
  $("dzPreview").hidden = false;
  $("clearBtn").hidden = false;
  $("sendBtn").disabled = false;
}

function clearFile() {
  state.file = null;
  $("fileInput").value = "";
  $("dzEmpty").hidden = false;
  $("dzPreview").hidden = true;
  $("clearBtn").hidden = true;
  $("sendBtn").disabled = true;
}

async function uploadFile(file) {
  const response = await fetch(`/api/upload?name=${encodeURIComponent(file.name)}`, { method: "POST", body: file });
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || "ההעלאה נכשלה");
  return body.file;
}

// ---------- Agent run ----------

function setStatus(kind, text) {
  $("agentStatus").dataset.state = kind;
  $("agentStatus").textContent = text;
}

function setRunning(running) {
  state.running = running;
  $("dropzone").classList.toggle("locked", running);
  $("sendBtn").disabled = running || !state.file;
  $("clearBtn").disabled = running;
  $("sendBtn").querySelector(".btn-spinner").hidden = !running;
  $("sendBtn").querySelector(".btn-label").textContent = running ? "האג'נט עובד…" : "שלחי לאג'נט";
  if (state.data) {
    renderPendingHint();
    renderTable(state.data.transactions);  // no corrections while the agent writes the same files
  }
}

async function startRun() {
  if (state.running) return;
  setRunning(true);
  state.newIds.clear();
  $("feed").replaceChildren();
  setStatus("working", "עובד");
  try {
    if (state.file) {
      const saved = await uploadFile(state.file);
      addStep("📥", "המסמך הועלה", saved, "step");
    }
  } catch (e) {
    addBubble("error", "⚠️", e.message);
    setStatus("error", "שגיאה");
    setRunning(false);
    return;
  }

  const socket = new WebSocket(`ws://${location.host}/ws/run`);
  let finished = false;
  socket.onmessage = (message) => handleEvent(JSON.parse(message.data), socket);
  socket.onclose = () => {
    if (!finished && state.running) {
      addBubble("error", "⚠️", "החיבור לשרת נותק. מה שנרשם עד עכשיו שמור.");
      setStatus("error", "נותק");
      endRun();
    }
  };
  socket.addEventListener("message", (m) => { if (JSON.parse(m.data).type === "done") finished = true; });
}

function endRun() {
  setRunning(false);
  clearFile();
  loadDashboard();
}

let pendingStep = null;  // the tool step waiting for its result

function handleEvent(event, socket) {
  switch (event.type) {
    case "thinking":
      addStep("💭", event.text, null, "step thought");
      break;
    case "tool_call": {
      const tool = TOOLS[event.name] || { icon: "🔧", label: event.name };
      pendingStep = addStep(tool.icon, tool.label, describeArgs(event.args), "step");
      pendingStep.querySelector(".step-title").append(el("span", "spinner"));
      break;
    }
    case "tool_result":
      showResult(event);
      break;
    case "ask":
      setStatus("asking", "מחכה לתשובה שלך");
      addQuestion(event, socket);
      break;
    case "final":
      addBubble("final", "✅", event.text);
      break;
    case "error":
      addBubble("error", "⚠️", event.text);
      break;
    case "done": {
      const failed = $("feed").querySelector(".step.error:last-child");
      setStatus(failed ? "error" : "done", failed ? "נעצר" : "סיים");
      endRun();
      break;
    }
  }
}

function describeArgs(args) {
  return Object.entries(args || {})
    .filter(([, v]) => v !== null && v !== "" && !Array.isArray(v))
    .map(([k, v]) => `${k}: ${String(v).length > 40 ? String(v).slice(0, 40) + "…" : v}`)
    .join(" · ");
}

function addStep(icon, title, args, className) {
  const feed = $("feed");
  const step = el("li", className);
  const body = el("div", "step-body");
  body.append(el("div", "step-title", title));
  if (args) body.append(el("div", "step-args", args));
  step.append(el("div", "step-icon", icon), body);
  feed.append(step);
  feed.scrollTop = feed.scrollHeight;
  return step;
}

function showResult(event) {
  const step = pendingStep || addStep("🔧", event.name, null, "step");
  pendingStep = null;
  step.querySelector(".spinner")?.remove();
  const body = step.querySelector(".step-body");
  body.append(el("div", `step-result ${event.ok ? "ok" : "fail"}`, `${event.ok ? "✓" : "✗"} ${event.text}`));
  for (const warning of event.warnings || []) body.append(el("p", "step-note warn", `⚠ ${warning}`));
  for (const note of event.info || []) body.append(el("p", "step-note info", `ℹ ${note}`));
  $("feed").scrollTop = $("feed").scrollHeight;

  if (event.name === "add_transaction" && event.ok) {
    // The dashboard updates live, on the month of what was just recorded
    state.newIds.add(event.transaction.id);
    state.month = event.transaction.date.slice(0, 7);
    loadDashboard();
  }
}

function addBubble(kind, icon, text) {
  const step = addStep(icon, "", null, `step ${kind}`);
  step.querySelector(".step-title").remove();
  const bubble = el("div", "bubble", text);
  step.querySelector(".step-body").append(bubble);
  $("feed").scrollTop = $("feed").scrollHeight;
  return bubble;
}

function addQuestion(event, socket) {
  const bubble = addBubble("question", "❓", event.question);
  const controls = [];

  const send = (answer, chosen) => {
    socket.send(JSON.stringify({ type: "answer", text: answer }));
    controls.forEach((c) => { c.disabled = true; });
    if (chosen) chosen.classList.add("chosen");
    bubble.append(el("div", "answered", answer ? `התשובה שלך: ${answer}` : "בלי תשובה"));
    form.remove();
    setStatus("working", "עובד");
  };

  if (event.options.length) {
    const options = el("div", "options");
    for (const text of event.options) {
      const button = el("button", "option", text);
      button.addEventListener("click", () => send(text, button));
      controls.push(button);
      options.append(button);
    }
    bubble.append(options);
  }

  const form = el("form", "free-answer");
  const input = el("input");
  input.placeholder = event.options.length ? "או תשובה אחרת…" : "התשובה שלך…";
  const submit = el("button", "btn btn-primary", "שליחה");
  submit.type = "submit";
  form.append(input, submit);
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    if (input.value.trim()) send(input.value.trim(), null);
  });
  controls.push(input, submit);
  bubble.append(form);
  if (!event.options.length) input.focus();
  $("feed").scrollTop = $("feed").scrollHeight;
}

// ---------- Theme ----------

function applyTheme(theme) {
  if (theme) document.documentElement.dataset.theme = theme;
  const dark = theme ? theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  $("themeToggle").textContent = dark ? "☀️" : "🌙";
}

function toggleTheme() {
  const dark = document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  const theme = dark ? "light" : "dark";
  try { localStorage.setItem("theme", theme); } catch { /* private window: the choice is not kept */ }
  applyTheme(theme);
  if (state.data) renderDashboard();  // charts read their colors from the theme
}

// ---------- Wiring ----------

function init() {
  let saved = null;
  try { saved = localStorage.getItem("theme"); } catch { /* ignore */ }
  applyTheme(saved);

  $("fileInput").addEventListener("change", (e) => chooseFile(e.target.files[0]));
  const zone = $("dropzone");
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("drag"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag");
    chooseFile(e.dataTransfer.files[0]);
  });
  // Pasting a screenshot (Ctrl+V) works too
  document.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (!item) return;
    const blob = item.getAsFile();
    const extension = blob.type.split("/")[1] === "jpeg" ? "jpg" : blob.type.split("/")[1];
    chooseFile(new File([blob], `צילום מסך ${new Date().toISOString().slice(0, 19).replace(/[T:]/g, "-")}.${extension}`, { type: blob.type }));
  });
  $("clearBtn").addEventListener("click", (e) => { e.preventDefault(); clearFile(); });
  $("sendBtn").addEventListener("click", startRun);
  $("prevMonth").addEventListener("click", () => shiftMonth(-1));
  $("nextMonth").addEventListener("click", () => shiftMonth(1));
  $("monthSelect").addEventListener("change", (e) => { state.month = e.target.value; loadDashboard(); });
  $("themeToggle").addEventListener("click", toggleTheme);
  $("imageModal").addEventListener("click", (e) => { if (e.target === $("imageModal")) $("imageModal").close(); });

  loadDashboard();
}

init();

/* Poll Review UI
 *
 * Loads multiple poll registries (e.g. poll_registry v1.json, poll_registry v2.json)
 * and lets the reviewer pick a preferred version per lesson. Selections are appended
 * to a reviewer-chosen CSV using the File System Access API (works on localhost).
 */

const $ = (id) => document.getElementById(id);

const TYPE_PALETTE = [
  "77, 163, 255",  // blue
  "80, 200, 200",  // teal
  "96, 214, 132",  // green
  "184, 215, 93",  // lime
  "244, 187, 80",  // amber
  "243, 122, 128", // coral
  "192, 138, 255", // purple
  "112, 212, 255", // sky
  "255, 156, 90",  // orange
  "140, 206, 255", // ice
  "255, 201, 108", // gold
  "166, 140, 255", // violet
  "120, 210, 170", // mint
  "255, 152, 182", // pink
];

const TYPE_COLOR_OVERRIDES = new Map([
  ["scenario_first_step", "77, 163, 255"],
  ["assumption_fragility", "80, 200, 200"],
  ["evidence_change_mind", "96, 214, 132"],
  ["design_constraint", "184, 215, 93"],
  ["counterfactual", "244, 187, 80"],
  ["stakeholder_lens", "243, 122, 128"],
  ["metric_to_track", "192, 138, 255"],
  ["implementation_risk", "112, 212, 255"],
  ["policy_instrument", "255, 156, 90"],
  ["tradeoff", "140, 206, 255"],
  ["failure_mode", "255, 201, 108"],
  ["sequence_order", "166, 140, 255"],
  ["classification_bucket", "120, 210, 170"],
]);

function hashString(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (h * 31 + s.charCodeAt(i)) >>> 0;
  }
  return h >>> 0;
}

function colorForType(typeId) {
  const t = (typeId ?? "").toString().toLowerCase().trim();
  if (!t) return "120, 130, 150"; // neutral for missing type
  if (TYPE_COLOR_OVERRIDES.has(t)) return TYPE_COLOR_OVERRIDES.get(t);
  const idx = hashString(t) % TYPE_PALETTE.length;
  return TYPE_PALETTE[idx];
}

const state = {
  registries: [], // [{label, path, items}]
  lessons: [],    // [{lesson_id, title, description, versions: [{label, entry}]}]
  idx: 0,
  csvHandle: null,
  csvPathLabel: "ready",
  decisions: new Map(), // lesson_id -> Map<label, {pollId, ts}>
  autoNext: true,
};

function csvEscape(s) {
  const t = (s ?? "").toString();
  if (/[",\n]/.test(t)) return `"${t.replaceAll('"', '""')}"`;
  return t;
}

function parseRegistryItems(json) {
  let items = json;
  if (items && typeof items === "object" && !Array.isArray(items) && Array.isArray(items.items)) {
    items = items.items;
  }
  if (!Array.isArray(items)) throw new Error("Registry is not a list");
  return items;
}

function labelFromFilename(path) {
  // polls/poll_registry v2.json -> v2
  const m = path.match(/v(\d+)\.json$/i);
  if (m) return `v${m[1]}`;
  if (path.endsWith("poll_registry.json")) return "base";
  return path.split("/").pop();
}

function lessonSortKey(lessonId) {
  // lesson ids like 6.1.2; sort numerically by dot parts
  const parts = (lessonId || "").split(".").map((x) => parseInt(x, 10));
  return parts.map((n) => (Number.isFinite(n) ? n : 9999));
}

function compareLessonId(a, b) {
  const ka = lessonSortKey(a);
  const kb = lessonSortKey(b);
  const n = Math.max(ka.length, kb.length);
  for (let i = 0; i < n; i++) {
    const va = ka[i] ?? 9999;
    const vb = kb[i] ?? 9999;
    if (va !== vb) return va - vb;
  }
  return (a || "").localeCompare(b || "");
}

async function tryLoadJson(path) {
  const res = await fetch(`../${encodeURI(path)}`);
  if (!res.ok) return null;
  return await res.json();
}

async function loadRegistries() {
  const path = "polls/poll_registry.json";
  const data = await tryLoadJson(path);
  if (!data) throw new Error("polls/poll_registry.json not found");

  const loaded = [];
  const label = labelFromFilename(path);
  const items = parseRegistryItems(data);
  loaded.push({ label, path, items });

  state.registries = loaded;
  $("pillFiles").textContent = `Registry: ${loaded.map((r) => r.label).join(", ") || "none"}`;
}

async function loadModuleStructure() {
  const data = await tryLoadJson("en/module_structure.json");
  if (!data) throw new Error("en/module_structure.json not found");
  return data;
}

function stripHtml(s) {
  return (s ?? "").toString().replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim();
}

function buildLessonIndex(moduleStructure) {
  const r = state.registries[0];
  if (!r) {
    state.lessons = [];
    return;
  }
  const byLesson = new Map();
  for (const it of r.items) {
    const lid = (it.lesson_id ?? "").toString();
    if (!lid) continue;
    if (!byLesson.has(lid)) byLesson.set(lid, []);
    byLesson.get(lid).push(it);
  }

  const infoByLesson = new Map();
  if (moduleStructure && Array.isArray(moduleStructure.modules)) {
    for (const mod of moduleStructure.modules) {
      for (const ch of (mod.chapters || [])) {
        for (const les of (ch.lessons || [])) {
          const lid = (les.id ?? "").toString();
          if (!lid) continue;
          infoByLesson.set(lid, {
            title: stripHtml(les.title ?? ""),
            description: stripHtml(les.description ?? ""),
          });
        }
      }
    }
  }

  const lessons = [];
  for (const lid of Array.from(byLesson.keys()).sort(compareLessonId)) {
    const entries = byLesson.get(lid) || [];
    const versions = entries.map((entry, i) => {
      const label = `v${i + 1}`;
      return { label, entry };
    });
    let title = "";
    let description = "";
    const info = infoByLesson.get(lid);
    if (info) {
      title = info.title || "";
      description = info.description || "";
    }
    lessons.push({ lesson_id: lid, title, description, versions });
  }
  state.lessons = lessons;
}

function updateKpi() {
  const done = state.decisions.size;
  const total = state.lessons.length;
  const pos = state.idx + 1;
  $("kpi").textContent = `Lesson ${pos}/${total} • Decided ${done}/${total}`;
}

function decisionForLesson(lid) {
  return state.decisions.get(lid) || new Map();
}

function render() {
  const lesson = state.lessons[state.idx];
  if (!lesson) return;

  $("lessonId").textContent = lesson.lesson_id;
  $("lessonTitle").textContent = lesson.title || "";
  const desc = lesson.description || "";
  $("lessonDesc").textContent = desc;
  $("lessonDesc").style.display = desc ? "block" : "none";

  const dec = decisionForLesson(lesson.lesson_id);
  const decLabels = Array.from(dec.keys());
  $("lessonStatus").textContent = decLabels.length ? `Selected: ${decLabels.join(", ")}` : "Not selected";
  $("btnPrev").disabled = state.idx <= 0;
  $("btnNext").disabled = state.idx >= state.lessons.length - 1;

  const cards = $("cards");
  cards.innerHTML = "";

  for (const v of lesson.versions) {
    const entry = v.entry;
    const card = document.createElement("div");
    card.className = "card";
    card.dataset.label = v.label;
    if (dec.has(v.label)) card.classList.add("selected");
    const arch = (entry.archetype_id ?? "").toString();
    card.style.setProperty("--card-rgb", colorForType(arch));

    const badge = document.createElement("div");
    badge.className = "badge";
    badge.textContent = v.label;

    const q = document.createElement("div");
    q.className = "q";
    q.textContent = (entry.question ?? "").toString();

    const meta = document.createElement("div");
    meta.className = "meta";
    const pollId = (entry.pollId ?? entry.poll_id ?? "").toString();
    if (pollId) {
      const s = document.createElement("span");
      s.textContent = `pollId: ${pollId}`;
      meta.appendChild(s);
    }
    if (arch) {
      const s = document.createElement("span");
      s.textContent = `type: ${arch}`;
      meta.appendChild(s);
    }
    const opener = (entry.opener_id ?? "").toString();
    if (opener) {
      const s = document.createElement("span");
      s.textContent = `opener: ${opener}`;
      meta.appendChild(s);
    }

    const opts = document.createElement("ol");
    const options = entry.options ?? [];
    const optColor = colorForType(arch);
    for (let i = 0; i < options.length; i++) {
      const o = options[i];
      const li = document.createElement("li");
      li.textContent = (o ?? "").toString();
      li.className = "opt";
      li.style.setProperty("--opt-rgb", optColor);
      // Inline styles to ensure visibility even if CSS isn't updating/cached.
      li.style.display = "block";
      li.style.padding = "7px 10px";
      li.style.borderRadius = "10px";
      li.style.border = `1px solid rgba(${optColor}, .35)`;
      li.style.borderLeft = `4px solid rgba(${optColor}, .85)`;
      li.style.background = `linear-gradient(90deg, rgba(${optColor}, .28), rgba(${optColor}, .08) 70%)`;
      opts.appendChild(li);
    }

    card.appendChild(badge);
    card.appendChild(q);
    if (meta.childNodes.length) card.appendChild(meta);
    card.appendChild(opts);

    card.addEventListener("click", async () => {
      await toggleVersion(lesson.lesson_id, v.label, entry);
    });

    cards.appendChild(card);
  }

  // Add a "none" option
  const noneCard = document.createElement("div");
  noneCard.className = "card";
  noneCard.dataset.label = "none";
  if (dec.has("none")) noneCard.classList.add("selected");

  const noneBadge = document.createElement("div");
  noneBadge.className = "badge";
  noneBadge.textContent = "none";

  const noneQ = document.createElement("div");
  noneQ.className = "q";
  noneQ.textContent = "None of these options should be selected for this lesson.";

  const noneMeta = document.createElement("div");
  noneMeta.className = "meta";
  const noneSpan = document.createElement("span");
  noneSpan.textContent = "No version selected";
  noneMeta.appendChild(noneSpan);

  noneCard.appendChild(noneBadge);
  noneCard.appendChild(noneQ);
  noneCard.appendChild(noneMeta);

  noneCard.addEventListener("click", async () => {
    await toggleVersion(lesson.lesson_id, "none", { pollId: "", question: "" });
  });

  cards.appendChild(noneCard);

  // Add an "I don't know" option
  const idkCard = document.createElement("div");
  idkCard.className = "card";
  idkCard.dataset.label = "idk";
  if (dec.has("idk")) idkCard.classList.add("selected");

  const idkBadge = document.createElement("div");
  idkBadge.className = "badge";
  idkBadge.textContent = "idk";

  const idkQ = document.createElement("div");
  idkQ.className = "q";
  idkQ.textContent = "I don't know which option is best for this lesson.";

  const idkMeta = document.createElement("div");
  idkMeta.className = "meta";
  const idkSpan = document.createElement("span");
  idkSpan.textContent = "No preference";
  idkMeta.appendChild(idkSpan);

  idkCard.appendChild(idkBadge);
  idkCard.appendChild(idkQ);
  idkCard.appendChild(idkMeta);

  idkCard.addEventListener("click", async () => {
    await toggleVersion(lesson.lesson_id, "idk", { pollId: "", question: "" });
  });

  cards.appendChild(idkCard);

  $("status").textContent = `Selections are stored locally until you download.`;

  updateKpi();
}

function toCsvRow({ lesson_id, chosen_version, pollId, question }) {
  return [
    csvEscape(lesson_id),
    csvEscape(chosen_version),
    csvEscape(pollId),
    csvEscape(question),
  ].join(",") + "\n";
}

function buildCsvFromSelections() {
  let out = "lesson_id,chosen_version,pollId,question\n";
  for (const [lessonId, m] of state.decisions.entries()) {
    for (const [label, meta] of m.entries()) {
      out += toCsvRow({
        lesson_id: lessonId,
        chosen_version: label,
        pollId: meta.pollId || "",
        question: meta.question || "",
      });
    }
  }
  return out;
}

function splitCsvLine(line) {
  const out = [];
  let cur = "";
  let inQ = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQ) {
      if (ch === '"' && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else if (ch === '"') {
        inQ = false;
      } else {
        cur += ch;
      }
    } else {
      if (ch === '"') inQ = true;
      else if (ch === ",") {
        out.push(cur);
        cur = "";
      } else {
        cur += ch;
      }
    }
  }
  out.push(cur);
  return out;
}

async function toggleVersion(lessonId, label, entry) {
  const pollId = (entry.pollId ?? entry.poll_id ?? "").toString();
  const question = (entry.question ?? "").toString();

  if (!state.decisions.has(lessonId)) state.decisions.set(lessonId, new Map());
  const m = state.decisions.get(lessonId);
  if (m.has(label)) {
    m.delete(label);
  } else {
    m.set(label, { pollId, question });
  }
  if (m.size === 0) state.decisions.delete(lessonId);

  render();
  if (state.autoNext) next();
}

function next() {
  state.idx = Math.min(state.idx + 1, state.lessons.length - 1);
  render();
}

function prev() {
  state.idx = Math.max(state.idx - 1, 0);
  render();
}

async function boot() {
  $("btnPrev").addEventListener("click", prev);
  $("btnNext").addEventListener("click", next);
  $("btnAutoNext").addEventListener("click", () => {
    state.autoNext = !state.autoNext;
    $("btnAutoNext").textContent = `Auto-advance: ${state.autoNext ? "ON" : "OFF"}`;
  });
  $("btnDownload").addEventListener("click", () => {
    const csv = buildCsvFromSelections();
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "poll_preferences.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  });

  await loadRegistries();
  const moduleStructure = await loadModuleStructure();
  buildLessonIndex(moduleStructure);
  updateKpi();
  render();
}

boot().catch((e) => {
  console.error(e);
  $("kpi").textContent = `Error: ${e.message || e}`;
  $("status").textContent = "Failed to load registries. Open devtools console for details.";
});

#!/usr/bin/env python3
import os, re, json, argparse, hashlib, time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, ValidationError, field_validator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Config + Models
# ----------------------------

LESSON_FILE_RE = re.compile(r"^(?P<m>\d+)\.(?P<c>\d+)\.(?P<l>-?\d+)\.json$")

class PollOption(BaseModel):
    id: int
    value: str = Field(min_length=3, max_length=180)

class PollContent(BaseModel):
    pollId: str
    title: str = Field(min_length=10, max_length=220)  # question
    description: str = Field(default="", max_length=280)  # optional short setup
    options: List[PollOption] = Field(min_length=4, max_length=4)
    labels: Dict[str, str]

    @field_validator("options")
    @classmethod
    def unique_options(cls, v):
        norm = [re.sub(r"\s+", " ", o.value.strip().lower()) for o in v]
        if len(set(norm)) != 4:
            raise ValueError("Options must be unique.")
        return v

class PollSegment(BaseModel):
    template_id: str = "poll"
    color_scheme: str = "light"
    content: PollContent

class RegistryEntry(BaseModel):
    pollId: str
    lesson_id: str
    question: str
    options: List[str]
    text_fingerprint: str
    # Optional metadata used for diversity controls (backwards compatible with older registries)
    archetype_id: Optional[str] = None
    opener_id: Optional[str] = None
    module: Optional[str] = None

# ----------------------------
# Utility
# ----------------------------

def log(msg: str) -> None:
    print(msg, flush=True)

def extract_json_object(text: str) -> str:
    txt = text.strip()
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if not m:
        raise ValueError("LLM did not return JSON object.")
    return m.group(0)

def safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # common fix: strip trailing commas
        cleaned = re.sub(r",\s*([}\]])", r"\1", text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            snippet = text.strip()
            if len(snippet) > 500:
                snippet = snippet[:500] + "...(truncated)"
            raise ValueError(f"LLM JSON parse failed: {e}. Raw: {snippet}")

def add_review_entry(module_logs: Dict[str, List[Dict[str, Any]]], module_name: str, entry: Dict[str, Any]) -> None:
    module_logs.setdefault(module_name, []).append(entry)

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

OPENERS = [
    ("which_factor_most", re.compile(r"^which factor most\b", re.I)),
    ("in_the_context_of", re.compile(r"^in the context of\b", re.I)),
    ("what_is_a_primary", re.compile(r"^what is a primary\b", re.I)),
    ("which_is_a_key", re.compile(r"^which is a key\b", re.I)),
    ("how_does", re.compile(r"^how does\b", re.I)),
    ("when_addressing", re.compile(r"^when addressing\b", re.I)),
    ("scenario_you_are", re.compile(r"^(you are|imagine you)\b", re.I)),
    ("metric_which_metric", re.compile(r"^which metric\b", re.I)),
    ("failure_mode", re.compile(r"^which failure\b|^what is the most likely failure\b", re.I)),
    ("counterfactual_if", re.compile(r"^if\b", re.I)),
]

HARD_AVOID_OPENERS = {
    # These are very common in the current registry and drive repetitive feel.
    "which_factor_most",
    "in_the_context_of",
}

HARD_AVOID_OPENER_EXAMPLES = [
    "Which factor most ...",
    "In the context of ...",
]

GENERIC_OPTION_PHRASES = [
    "availability of",
    "public acceptance",
    "government policies",
    "grid infrastructure",
    "existing infrastructure",
    "upfront costs",
    "cost disparities",
]

ARCHETYPES: List[Dict[str, Any]] = [
    {
        "id": "scenario_first_step",
        "style_tag": "scenario",
        "prompt": "Write a scenario-based question where the learner must choose the FIRST action/decision. Options must be actionable verbs and non-overlapping.",
        "question_starts": ["You are advising", "Imagine you are", "A ministry asks you to"],
        "option_shape": "4 actions (verbs), each a distinct first step.",
    },
    {
        "id": "tradeoff",
        "style_tag": "tradeoff",
        "prompt": "Write a trade-off question where none of the options is obviously correct. Options must each represent a different trade-off dimension.",
        "question_starts": ["Which trade-off would you accept first", "Which compromise is most acceptable"],
        "option_shape": "4 trade-offs (e.g., cost vs speed vs equity vs reliability) tailored to lesson.",
    },
    {
        "id": "metric_to_track",
        "style_tag": "diagnostic",
        "prompt": "Ask which metric/indicator should be tracked to evaluate progress or detect problems. Options must be measurable indicators, not actions.",
        "question_starts": ["Which metric would you track", "Which indicator best signals"],
        "option_shape": "4 metrics/indicators; each distinct and measurable.",
    },
    {
        "id": "failure_mode",
        "style_tag": "prediction",
        "prompt": "Ask about the most likely failure mode / unintended consequence when implementing or scaling something in the lesson. Options must be distinct failure modes.",
        "question_starts": ["What is the most likely failure mode", "Which failure is most likely"],
        "option_shape": "4 failure modes; each distinct and plausible.",
    },
    {
        "id": "stakeholder_lens",
        "style_tag": "values",
        "prompt": "Ask which stakeholder would most support or oppose a proposed change. Options must be stakeholder groups with distinct incentives (not generic).",
        "question_starts": ["Which stakeholder would be most concerned", "Who would push back most"],
        "option_shape": "4 stakeholder groups; each distinct.",
    },
    {
        "id": "assumption_fragility",
        "style_tag": "diagnostic",
        "prompt": "Ask which assumption is most fragile / most likely to break in practice. Options must be assumptions (not actions).",
        "question_starts": ["Which assumption is most fragile", "Which assumption is riskiest"],
        "option_shape": "4 assumptions; each distinct.",
    },
    {
        "id": "sequence_order",
        "style_tag": "prioritization",
        "prompt": "Ask what should happen BEFORE something else (sequence / prerequisites). Options must be prerequisites/sequence steps.",
        "question_starts": ["What should come before", "Which prerequisite matters most before"],
        "option_shape": "4 prerequisite steps; each distinct.",
    },
    {
        "id": "evidence_change_mind",
        "style_tag": "values",
        "prompt": "Ask what evidence would most change a decision or belief. Options must be evidence types or findings, not actions.",
        "question_starts": ["What evidence would change your mind", "Which finding would most change"],
        "option_shape": "4 evidence types/findings; each distinct.",
    },
    {
        "id": "counterfactual",
        "style_tag": "prediction",
        "prompt": "Ask a counterfactual 'If X changed, what would happen first?' tied to lesson content. Options must be plausible outcomes, not generic.",
        "question_starts": ["If one constraint disappeared", "If a key assumption changed", "If funding were guaranteed"],
        "option_shape": "4 outcomes; each distinct and plausible.",
    },
    {
        "id": "policy_instrument",
        "style_tag": "prioritization",
        "prompt": "Ask which policy instrument is the best fit for a clearly stated goal in the lesson. Options must be distinct instrument types.",
        "question_starts": ["Which policy instrument best fits", "Which tool would you choose to"],
        "option_shape": "4 policy tools/instruments; each distinct (e.g., subsidy, standards, auctions, disclosure).",
    },
    {
        "id": "implementation_risk",
        "style_tag": "tradeoff",
        "prompt": "Ask which implementation risk should be mitigated first. Options must be risks (not actions), and should be lesson-specific.",
        "question_starts": ["Which risk would you mitigate first", "Which implementation risk is most urgent"],
        "option_shape": "4 risks; each distinct.",
    },
    {
        "id": "classification_bucket",
        "style_tag": "diagnostic",
        "prompt": "Ask the learner to classify a situation described in the lesson into one of four categories. Options must be mutually exclusive category labels.",
        "question_starts": ["Which category best describes", "This situation is best classified as"],
        "option_shape": "4 category labels; mutually exclusive.",
    },
    {
        "id": "design_constraint",
        "style_tag": "values",
        "prompt": "Ask which design constraint should be non-negotiable (equity, reliability, affordability, transparency, etc.) given the lesson. Options must be constraints, not actions.",
        "question_starts": ["Which constraint should be non-negotiable", "Which requirement must be locked in"],
        "option_shape": "4 constraints; each distinct.",
    },
]

def opener_id_for_question(q: str) -> str:
    qn = normalize_ws(q).lower()
    for oid, pat in OPENERS:
        if pat.search(qn):
            return oid
    return "other"

def choose_anchor_terms(card: "LessonCard", max_terms: int = 6) -> List[str]:
    # Prefer short, concrete phrases from key_concepts/takeaways/excerpts.
    raw: List[str] = []
    for s in (card.key_concepts or [])[:6]:
        raw.append(s)
    for s in (card.key_takeaways or [])[:6]:
        raw.append(s)
    for s in (card.excerpts or [])[:2]:
        raw.append(s)

    out: List[str] = []
    seen = set()
    for s in raw:
        s2 = strip_html(str(s))
        s2 = normalize_ws(s2)
        if not s2:
            continue
        # take first clause-ish, keep it short
        s2 = re.split(r"[.;:–—-]", s2)[0]
        s2 = normalize_ws(s2)
        if len(s2) < 8:
            continue
        if len(s2) > 70:
            s2 = s2[:70].rsplit(" ", 1)[0]
        key = s2.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s2)
        if len(out) >= max_terms:
            break
    return out

def diversity_check(
    question: str,
    options: List[str],
    anchor_terms: List[str],
    attempt: int,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    reasons: List[str] = []
    meta: Dict[str, Any] = {}

    oid = opener_id_for_question(question)
    meta["opener_id"] = oid
    if oid in HARD_AVOID_OPENERS:
        reasons.append(f"Avoid opener '{oid}' (too repetitive); use a different format.")

    qn = normalize_ws(question).lower()
    anchors_hit_q = [a for a in anchor_terms if a.lower() in qn]
    anchors_hit_opts = []
    for a in anchor_terms:
        al = a.lower()
        if any(al in normalize_ws(o).lower() for o in options):
            anchors_hit_opts.append(a)
    anchors_hit_opts = list(dict.fromkeys(anchors_hit_opts))
    meta["anchors_hit_q"] = anchors_hit_q
    meta["anchors_hit_opts"] = anchors_hit_opts

    # Progressively enforce anchoring on later attempts.
    if attempt >= 2 and len(anchors_hit_q) < 1:
        reasons.append("Question is too generic; include at least 1 anchor term from the lesson card.")
    if attempt >= 3 and len(anchors_hit_opts) < 2:
        reasons.append("Options feel generic; include at least 2 distinct anchor terms across the options.")

    # Avoid generic option filler patterns (they dominated earlier outputs).
    bad_phrase_hits = []
    for o in options:
        ol = normalize_ws(o).lower()
        for ph in GENERIC_OPTION_PHRASES:
            if ph in ol:
                bad_phrase_hits.append(ph)
    meta["generic_option_phrases"] = sorted(set(bad_phrase_hits))
    if attempt >= 2 and len(set(bad_phrase_hits)) >= 2:
        reasons.append(f"Options reuse generic phrases {sorted(set(bad_phrase_hits))}; rewrite with lesson-specific language.")

    return (len(reasons) == 0), reasons, meta

def load_existing_review_counts(review_dir: str) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    # Returns (module->archetype counts, module->opener counts)
    archetype_counts: Dict[str, Dict[str, int]] = {}
    opener_counts: Dict[str, Dict[str, int]] = {}
    if not os.path.isdir(review_dir):
        return archetype_counts, opener_counts
    for fn in os.listdir(review_dir):
        if not fn.endswith(".json"):
            continue
        module = os.path.splitext(fn)[0]
        try:
            items = read_json(os.path.join(review_dir, fn))
        except Exception:
            continue
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            aid = str(it.get("archetype_id") or "")
            oid = str(it.get("opener_id") or "")
            if aid:
                archetype_counts.setdefault(module, {})[aid] = archetype_counts.setdefault(module, {}).get(aid, 0) + 1
            if oid:
                opener_counts.setdefault(module, {})[oid] = opener_counts.setdefault(module, {}).get(oid, 0) + 1
    return archetype_counts, opener_counts

def pick_archetype_for_attempt(
    module_name: str,
    seed: str,
    tried: set,
    module_archetype_counts: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    # Pick a least-used archetype within the module, avoiding repeats for the same lesson.
    counts = module_archetype_counts.get(module_name, {})
    candidates = [a for a in ARCHETYPES if a["id"] not in tried] or ARCHETYPES[:]
    min_count = min(counts.get(a["id"], 0) for a in candidates)
    least = [a for a in candidates if counts.get(a["id"], 0) == min_count]
    least.sort(key=lambda a: a["id"])
    # Deterministic shuffle within the least-used set for better spread.
    idx = int(stable_hash(seed), 16) % len(least)
    return least[idx]

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def strip_html(txt: str) -> str:
    if not txt:
        return ""
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def segment_text(seg: Dict[str, Any]) -> str:
    c = seg.get("content", {})
    if isinstance(c, dict):
        # common keys
        for k in ["text", "body", "title", "description", "intro"]:
            v = c.get(k)
            if isinstance(v, str) and v.strip():
                return v
        # nested patterns
        if "items" in c and isinstance(c["items"], list):
            parts=[]
            for it in c["items"]:
                if isinstance(it, dict):
                    for kk in ["title","body","text","description"]:
                        vv=it.get(kk)
                        if isinstance(vv,str) and vv.strip():
                            parts.append(vv)
            return " ".join(parts)
    return ""

def has_poll_segment(segments: List[Dict[str, Any]]) -> bool:
    return any(isinstance(s, dict) and s.get("template_id") == "poll" for s in segments)

def stable_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def build_poll_segment(poll_id: str, question: str, desc: str, options: List[str]) -> Dict[str, Any]:
    seg = {
        "template_id": "poll",
        "color_scheme": "light",
        "content": {
            "pollId": poll_id,
            "title": question,
            "description": desc or "",
            "options": [{"id": i+1, "value": options[i]} for i in range(4)],
            "labels": {"submit":"Submit","cancel":"Cancel","edit":"Edit your response","votingAs":"Voting as"}
        }
    }
    # validate
    PollSegment.model_validate(seg)
    return seg

MAX_QUESTION_LEN = 220
MAX_DESC_LEN = 280
MAX_OPTION_LEN = 180

def insert_poll_segment(lesson: Dict[str, Any], poll_segment: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    # insert before connection_next if present, else near end.
    segs = lesson.get("segments", [])
    if not isinstance(segs, list):
        raise ValueError("Lesson JSON missing segments list.")
    idx = next((i for i,s in enumerate(segs) if isinstance(s, dict) and s.get("template_id") == "connection_next"), None)
    if idx is None:
        idx = len(segs)
    segs2 = segs[:idx] + [poll_segment] + segs[idx:]
    lesson2 = dict(lesson)
    lesson2["segments"] = segs2
    return lesson2, idx

# ----------------------------
# Lesson card extraction
# ----------------------------

@dataclass
class LessonCard:
    lesson_id: str
    title: str
    module: str
    chapter: str
    key_concepts: List[str]
    key_takeaways: List[str]
    key_resources: List[str]
    excerpts: List[str]

def extract_lesson_title(lesson: Dict[str, Any]) -> str:
    for seg in lesson.get("segments", []):
        if isinstance(seg, dict) and seg.get("template_id") == "lesson_cover":
            c = seg.get("content", {})
            if isinstance(c, dict):
                for k in ["title", "label", "intro", "text"]:
                    v = c.get(k)
                    if isinstance(v, str) and v.strip():
                        # title often in title
                        if k == "title":
                            return strip_html(v)
                # sometimes title stored differently
    # fallback: id
    return str(lesson.get("id","")).strip()

def extract_items(seg: Dict[str, Any]) -> List[str]:
    c = seg.get("content", {})
    out=[]
    if isinstance(c, dict) and isinstance(c.get("items"), list):
        for it in c["items"]:
            if isinstance(it, dict):
                t = it.get("title") or it.get("text") or ""
                b = it.get("body") or it.get("description") or ""
                s = " - ".join([strip_html(x) for x in [t,b] if isinstance(x,str) and x.strip()])
                if s:
                    out.append(s)
    return out

def extract_excerpts(lesson: Dict[str, Any], max_excerpts: int = 2, min_len: int = 80) -> List[str]:
    texts=[]
    for seg in lesson.get("segments", []):
        if not isinstance(seg, dict): 
            continue
        if seg.get("template_id") in {"text","quote_large_with_name","quote_large_without_name","paragraph_large","paragraph_medium","paragraph_small"}:
            t=strip_html(segment_text(seg))
            if len(t) >= min_len:
                texts.append(t)
        elif seg.get("template_id") in {"key_concepts","key_takeaways"}:
            for it in extract_items(seg):
                if len(it) >= min_len:
                    texts.append(it)
    # pick distinct by simple hash of tail
    uniq=[]
    seen=set()
    for t in texts:
        h=stable_hash(t[-180:])
        if h in seen: 
            continue
        seen.add(h)
        uniq.append(t)
        if len(uniq) >= max_excerpts:
            break
    return uniq

def build_lesson_card(path: str, lesson: Dict[str, Any]) -> LessonCard:
    lid = str(lesson.get("id","")).strip()
    fname = os.path.basename(path)
    m = LESSON_FILE_RE.match(fname)
    module = m.group("m") if m else ""
    chapter = m.group("c") if m else ""
    title = extract_lesson_title(lesson)
    key_concepts=[]
    key_takeaways=[]
    key_resources=[]
    for seg in lesson.get("segments", []):
        if not isinstance(seg, dict): 
            continue
        tid = seg.get("template_id")
        if tid == "key_concepts":
            key_concepts = extract_items(seg)[:8]
        elif tid == "key_takeaways":
            key_takeaways = extract_items(seg)[:8]
        elif tid == "key_resources":
            key_resources = extract_items(seg)[:8]
    excerpts = extract_excerpts(lesson)
    return LessonCard(
        lesson_id=lid or f"{module}.{chapter}",
        title=title,
        module=module,
        chapter=chapter,
        key_concepts=key_concepts,
        key_takeaways=key_takeaways,
        key_resources=key_resources,
        excerpts=excerpts
    )

# ----------------------------
# Dedup registry (TF-IDF)
# ----------------------------

def poll_text(question: str, options: List[str]) -> str:
    return (question.strip() + " || " + " | ".join([o.strip() for o in options])).strip()

def registry_texts(registry: List[RegistryEntry]) -> List[str]:
    return [poll_text(r.question, r.options) for r in registry]

def is_too_similar(candidate_text: str, registry: List[RegistryEntry], threshold: float) -> Tuple[bool, float]:
    if not registry:
        return False, 0.0
    texts = registry_texts(registry) + [candidate_text]
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=8000)
    X = vec.fit_transform(texts)
    sims = cosine_similarity(X[-1], X[:-1])[0]
    best = float(sims.max()) if sims.size else 0.0
    return best >= threshold, best

# ----------------------------
# LLM adapter
# ----------------------------

def get_llm_client():
    # OpenAI standard
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        return ("openai", OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
    # Azure OpenAI
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_DEPLOYMENT"):
        from openai import AzureOpenAI
        return ("azure", AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION","2024-08-01-preview")
        ))
    return (None, None)

def llm_generate_poll(
    card: LessonCard,
    registry: List[RegistryEntry],
    attempt: int,
    model_hint: Optional[str]=None,
    previous: Optional[Dict[str, Any]] = None,
    feedback: Optional[List[str]] = None,
    archetype: Optional[Dict[str, Any]] = None,
    anchor_terms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    mode, client = get_llm_client()
    if client is None:
        raise RuntimeError("No LLM configured. Set OPENAI_API_KEY (or Azure env vars) before running.")
    feedback = feedback or []
    avoid = [r.question for r in registry[-200:]]  # last 200 for token control
    avoid_opts = [opt for r in registry[-200:] for opt in r.options]
    avoid_block = "\n".join([f"- {q}" for q in avoid[:80]])  # cap
    avoid_opts_block = "\n".join([f"- {o}" for o in avoid_opts[:120]])

    system = (
        "You generate one high-quality interactive poll per lesson for a global Sustainable Energy Academy. "
        "Polls must be engaging, specific to lesson content, neutral in tone, and have exactly 4 mutually exclusive answer options. "
        "Avoid repetition across lessons."
    )
    user = f"""LESSON CARD
- Lesson ID: {card.lesson_id}
- Lesson title: {card.title}
- Module: {card.module}  Chapter: {card.chapter}
- Key concepts: {card.key_concepts[:6]}
- Key takeaways: {card.key_takeaways[:6]}
- Key resources: {card.key_resources[:4]}
- Excerpts: {card.excerpts[:2]}

CONSTRAINTS
- Produce EXACTLY 1 poll.
- Question: 1 sentence (max ~160 chars). Can be short or scenario-based.
- Provide 4 answer options. Each option must be distinct and not overlap.
- Options can be different lengths; avoid "All of the above" and avoid obvious right answers.
- Keep language globally appropriate, not country-specific unless lesson indicates it.
- Do NOT repeat questions or options similar to these:
QUESTIONS TO AVOID (examples):
{avoid_block}

OPTIONS TO AVOID (examples):
{avoid_opts_block}

FORMAT DIVERSITY
- Do NOT start the question with these overused openers: {HARD_AVOID_OPENER_EXAMPLES}
- Avoid generic option phrases like: {GENERIC_OPTION_PHRASES}

ARCHETYPE
{json.dumps(archetype or {}, ensure_ascii=False)}
- Follow the archetype prompt. If the archetype provides question_starts, prefer one of them.

ANCHOR TERMS (use these to stay specific to the lesson)
{anchor_terms or []}
- Include at least 1 anchor term in the question where natural.
- Try to use at least 2 distinct anchor terms across the options.

QUALITY CHECKLIST (aim for 3+/5 on each)
- Specific to lesson concepts
- Options are clearly distinct and mutually exclusive
- Neutral tone (no right answer)
- Engaging and clear
- Output must be valid JSON (double quotes, no trailing commas) with no extra text.

OUTPUT FORMAT (JSON only)
{{
  "question": "...",
  "description": "optional short setup (0-180 chars) or empty string",
  "options": ["...", "...", "...", "..."],
  "style_tag": "scenario|tradeoff|prediction|prioritization|values|diagnostic"
}}
"""

    if previous:
        user += "\nREVISION CONTEXT\nPrevious poll JSON:\n" + json.dumps(previous, ensure_ascii=False)
    if feedback:
        user += "\nReviewer feedback:\n" + "\n".join([f"- {f}" for f in feedback])
        user += "\nRevise the poll to address the feedback. You may replace question/options entirely if needed."

    if mode == "openai":
        model = model_hint or os.getenv("OPENAI_MODEL","gpt-4.1-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.9 if attempt == 1 else 0.7,
        )
        txt = resp.choices[0].message.content
    else:
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.9 if attempt == 1 else 0.7,
        )
        txt = resp.choices[0].message.content

    # extract JSON
    txt = txt.strip()
    obj_text = extract_json_object(txt)
    obj = safe_json_loads(obj_text)
    if not isinstance(obj, dict):
        raise ValueError("LLM JSON is not an object.")
    return obj

def llm_critic(card: LessonCard, poll_obj: Dict[str, Any], archetype: Optional[Dict[str, Any]] = None, anchor_terms: Optional[List[str]] = None) -> Dict[str, Any]:
    mode, client = get_llm_client()
    if client is None:
        raise RuntimeError("No LLM configured for critic pass.")
    system = "You are a strict reviewer for curriculum polls. You score quality and flag issues."
    user = f"""CRITIC PLAN v2
Score each criterion 0-5 with a brief note.
Scoring guide:
- 0-1: missing/invalid
- 2: weak
- 3: acceptable
- 4: good
- 5: excellent

Criteria definitions:
- specificity: Clearly anchored to lesson concepts or excerpts (not generic).
- distinctness: Options are clearly different and do not overlap.
- separability: Options are mutually exclusive and individually selectable.
- neutrality: No leading language or obvious correct answer.
- engagement: Thought-provoking or scenario-based; not trivial.
- clarity: Wording is unambiguous and concise.

overall_pass must be true ONLY if all scores >= 3 and there are no critical issues.
Provide 1-3 fix_suggestions that are concrete edits (e.g., rephrase question, replace option X).

LESSON: {card.title}
CARD CONCEPTS: {card.key_concepts[:6]}
ARCHETYPE: {json.dumps(archetype or {}, ensure_ascii=False)}
ANCHOR TERMS: {anchor_terms or []}
POLL:
Question: {poll_obj.get("question")}
Description: {poll_obj.get("description")}
Options: {poll_obj.get("options")}

Return JSON only:
{{
  "specificity": {{"score":0,"note":""}},
  "distinctness": {{"score":0,"note":""}},
  "separability": {{"score":0,"note":""}},
  "neutrality": {{"score":0,"note":""}},
  "engagement": {{"score":0,"note":""}},
  "clarity": {{"score":0,"note":""}},
  "overall_pass": true,
  "fix_suggestions": ["..."]
}}
"""
    if mode == "openai":
        model = os.getenv("OPENAI_MODEL","gpt-4.1-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.2,
        )
        txt = resp.choices[0].message.content
    else:
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.2,
        )
        txt = resp.choices[0].message.content
    obj_text = extract_json_object(txt)
    return safe_json_loads(obj_text)

CRITIC_KEYS = ["specificity","distinctness","separability","neutrality","engagement","clarity"]
CRITIC_PLAN_VERSION = "2026-02-10-v2"

def critic_summary(crit: Dict[str, Any], min_score: int = 3) -> Tuple[Dict[str, int], List[str], Dict[str, str]]:
    scores = {k: int(crit.get(k, {}).get("score", 0)) for k in CRITIC_KEYS}
    notes = {k: str(crit.get(k, {}).get("note", "")).strip() for k in CRITIC_KEYS}
    failed_criteria = [k for k, v in scores.items() if v < min_score]
    return scores, failed_criteria, notes

def passes_critic(crit: Dict[str, Any], min_score: int = 3) -> bool:
    try:
        scores, failed_criteria, _ = critic_summary(crit, min_score)
        return len(failed_criteria) == 0
    except Exception:
        return False

# ----------------------------
# Main generation
# ----------------------------

def discover_lesson_files(root: str) -> List[str]:
    paths=[]
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith(".json"):
                continue
            m = LESSON_FILE_RE.match(fn)
            if not m:
                continue
            # lesson number filter: generate for real lessons only (>=1)
            try:
                l = int(m.group("l"))
            except:
                continue
            if l < 1:
                continue
            paths.append(os.path.join(dirpath, fn))
    return sorted(paths)

def load_registry(path: str) -> List[RegistryEntry]:
    if not os.path.exists(path):
        return []
    raw = read_json(path)
    if isinstance(raw, dict) and "items" in raw:
        raw = raw["items"]
    out=[]
    for r in raw:
        try:
            out.append(RegistryEntry.model_validate(r))
        except ValidationError:
            continue
    return out

def save_registry(path: str, reg: List[RegistryEntry]) -> None:
    write_json(path, [r.model_dump() for r in reg])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lessons_root", required=True, help="Path to /en folder containing Module_* subfolders")
    ap.add_argument("--output_root", required=True, help="Output folder for updated lessons + exports")
    ap.add_argument("--registry_path", default=None, help="Path to poll_registry.json (defaults under output_root)")
    ap.add_argument("--target_per_lesson", type=int, default=1)
    ap.add_argument("--max_attempts", type=int, default=4)
    ap.add_argument("--similarity_threshold", type=float, default=0.86)
    ap.add_argument("--print_polls", action="store_true", help="Print each poll as it is generated")
    ap.add_argument("--debug_failures", action="store_true", help="Print per-lesson failures and last error")
    ap.add_argument("--critic_min_score", type=int, default=3, help="Minimum per-criterion critic score to accept a poll")
    ap.add_argument("--max_lessons", type=int, default=0, help="If >0, process only the first N lesson files (for quick testing)")
    ap.add_argument("--overwrite_existing_polls", action="store_true")
    args = ap.parse_args()

    lessons_root = os.path.abspath(args.lessons_root)
    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    registry_path = args.registry_path or os.path.join(output_root, "poll_registry.json")

    lesson_files = discover_lesson_files(lessons_root)
    if not lesson_files:
        raise RuntimeError(f"No lesson files found under: {lessons_root}. Expected files like 2.1.3.json inside Module_* folders.")

    registry = load_registry(registry_path)
    total_lessons = len(lesson_files)
    log(f"✅ Found {total_lessons} lesson files under {lessons_root}")
    log(f"✅ Loaded registry entries: {len(registry)}")
    global_opener_counts: Dict[str, int] = {}
    for r in registry:
        oid = opener_id_for_question(r.question)
        global_opener_counts[oid] = global_opener_counts.get(oid, 0) + 1
    # Preflight LLM config (fail fast instead of silently skipping)
    mode, client = get_llm_client()
    if client is None:
        raise RuntimeError(
            "No LLM configured. Set OPENAI_API_KEY (and optionally OPENAI_MODEL) "
            "or set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT "
            "(and optionally AZURE_OPENAI_API_VERSION) before running."
        )

    review_dir = os.path.join(output_root, "review_logs")
    review_map_path = os.path.join(output_root, "review_exports", "polls_en_map.json")

    generated = 0
    accepted_best = 0
    skipped_registry = 0
    skipped_existing = 0
    mapping = {}
    failed_lessons = 0
    module_logs: Dict[str, List[Dict[str, Any]]] = {}

    module_archetype_counts, module_opener_counts = load_existing_review_counts(review_dir)

    if args.max_lessons and args.max_lessons > 0:
        lesson_files = lesson_files[: args.max_lessons]
        total_lessons = len(lesson_files)
        log(f"🧪 max_lessons enabled: processing first {total_lessons} lessons")

    for idx, path in enumerate(lesson_files, start=1):
        lesson = read_json(path)
        lid = str(lesson.get("id","")).strip() or os.path.splitext(os.path.basename(path))[0]
        rel = os.path.relpath(path, lessons_root)
        module_folder = rel.split(os.sep)[0] if rel else "Module_unknown"
        log(f"▶️ [{idx}/{total_lessons}] Processing {lid} ({rel})")

        if (not args.overwrite_existing_polls) and has_poll_segment(lesson.get("segments", [])):
            skipped_existing += 1
            log("  ⏭️ Skipped (already has poll segment)")
            continue

        card = build_lesson_card(path, lesson)
        anchor_terms = choose_anchor_terms(card)

        # Generate one poll
        poll_obj = None
        critic = None
        last_err = None
        prev_poll = None
        feedback = None
        best_candidate = None
        best_score_key = None
        tried_archetypes: set = set()
        for attempt in range(1, args.max_attempts + 1):
            try:
                archetype = pick_archetype_for_attempt(module_folder, f"{module_folder}:{lid}:{attempt}", tried_archetypes, module_archetype_counts)
                tried_archetypes.add(archetype["id"])
                log(f"  🔁 Attempt {attempt}/{args.max_attempts} generating poll...")
                log(f"  🧩 Archetype: {archetype['id']}")
                t0 = time.time()
                if feedback:
                    log(f"  🛠️ Using critic feedback ({len(feedback)} items) for retry")
                raw = llm_generate_poll(
                    card,
                    registry,
                    attempt,
                    previous=prev_poll,
                    feedback=feedback,
                    archetype=archetype,
                    anchor_terms=anchor_terms,
                )
                log(f"  ⏱️ LLM generate took {time.time() - t0:.1f}s")
                question = str(raw.get("question","")).strip()
                desc = str(raw.get("description","") or "").strip()
                opts = raw.get("options", [])
                if not (question and isinstance(opts, list) and len(opts) == 4):
                    raise ValueError("Bad structure: need question and 4 options.")
                opts = [str(o).strip() for o in opts]
                # enforce hard limits to avoid late Pydantic failures and to keep reviewer UX consistent
                len_reasons = []
                if len(question) > MAX_QUESTION_LEN:
                    len_reasons.append(f"Question too long ({len(question)}>{MAX_QUESTION_LEN}); shorten to <= {MAX_QUESTION_LEN} chars.")
                if len(desc) > MAX_DESC_LEN:
                    # Description is optional; truncation is safe and avoids needless retries.
                    desc = desc[:MAX_DESC_LEN].rstrip()
                too_long_opts = [o for o in opts if len(o) > MAX_OPTION_LEN]
                if too_long_opts:
                    len_reasons.append(f"One or more options too long (>{MAX_OPTION_LEN} chars); shorten options.")
                if len_reasons:
                    log(f"  ⚠️ Length gate failed; retrying. ({len(len_reasons)} reasons)")
                    if args.debug_failures:
                        for r in len_reasons:
                            log(f"    - {r}")
                    feedback = (feedback or []) + len_reasons
                    prev_poll = {"question": question, "description": desc, "options": opts}
                    last_err = "Length gate failed"
                    continue
                # quick hygiene
                if any(len(o) < 3 for o in opts):
                    raise ValueError("Option too short.")
                if len(set([re.sub(r"\s+"," ",o.lower()) for o in opts])) != 4:
                    raise ValueError("Options not unique.")
                cand_text = poll_text(question, opts)
                too_sim, best = is_too_similar(cand_text, registry, args.similarity_threshold)
                if too_sim:
                    skipped_registry += 1
                    last_err = f"Too similar to registry (best={best:.3f})."
                    log(f"  ⏭️ Too similar to registry (best={best:.3f})")
                    continue

                div_ok, div_reasons, div_meta = diversity_check(question, opts, anchor_terms, attempt)
                opener_id = div_meta.get("opener_id", opener_id_for_question(question))
                module_opener_counts.setdefault(module_folder, {})
                # If this opener is already overused in this module, push back harder.
                if module_opener_counts[module_folder].get(opener_id, 0) >= 4 and opener_id != "other":
                    div_ok = False
                    div_reasons = div_reasons + [f"Opener '{opener_id}' is already common in {module_folder}; use a different opening."]
                # If this opener is already overused globally (across runs), push back.
                if global_opener_counts.get(opener_id, 0) >= 15 and opener_id != "other":
                    div_ok = False
                    div_reasons = div_reasons + [f"Opener '{opener_id}' is already common globally; use a different opening."]
                if not div_ok:
                    log(f"  🎭 Diversity gate failed (will still score candidate). ({len(div_reasons)} reasons)")
                    if args.debug_failures:
                        for r in div_reasons[:6]:
                            log(f"    - {r}")
                    # Feed back into next attempt.
                    feedback = (feedback or []) + div_reasons
                    prev_poll = {"question": question, "description": desc, "options": opts}
                    last_err = "Diversity gate failed"

                log("  🔎 Running critic...")
                t1 = time.time()
                critic = llm_critic(card, {"question":question,"description":desc,"options":opts}, archetype=archetype, anchor_terms=anchor_terms)
                log(f"  ⏱️ Critic took {time.time() - t1:.1f}s")
                scores, failed_criteria, notes = critic_summary(critic, args.critic_min_score)
                # Prefer higher total score, fewer failures, and fewer diversity issues.
                div_penalty = len(div_reasons)
                score_key = (sum(scores.values()) - 2 * div_penalty, -len(failed_criteria) - div_penalty)
                candidate = {
                    "question": question,
                    "description": desc,
                    "options": opts,
                    "scores": scores,
                    "failed_criteria": failed_criteria,
                    "notes": notes,
                    "critic": critic,
                    "attempt": attempt,
                    "style_tag": str(raw.get("style_tag","")).strip(),
                    "archetype_id": archetype["id"],
                    "opener_id": opener_id,
                    "anchor_terms": anchor_terms,
                    "diversity_ok": div_ok,
                    "diversity_reasons": div_reasons,
                    "diversity_meta": div_meta,
                }
                if best_score_key is None or score_key > best_score_key:
                    best_score_key = score_key
                    best_candidate = candidate

                if not passes_critic(critic, args.critic_min_score):
                    last_err = f"Critic failed: {scores}"
                    log(f"  ⚠️ Critic failed (scores={scores}, failed={failed_criteria}); retrying.")
                    # prepare feedback for next attempt
                    feedback = critic.get("fix_suggestions") or []
                    if not feedback:
                        feedback = [f"{k}: {notes[k]}" for k in failed_criteria if notes.get(k)]
                    prev_poll = {"question": question, "description": desc, "options": opts}
                    if args.debug_failures:
                        if notes:
                            log(f"  Notes: { {k: notes[k] for k in failed_criteria if notes.get(k)} }")
                        log(f"  Candidate: Q: {question}")
                        log(f"  Options: {opts}")
                    continue

                if not div_ok:
                    log("  ⚠️ Candidate passed critic but failed diversity; retrying.")
                    continue

                poll_id = f"{lid}__poll__{stable_hash(cand_text)}"
                poll_seg = build_poll_segment(poll_id, question, desc, opts)

                review_entry = {
                    "lesson_id": lid,
                    "lesson_title": card.title,
                    "module": module_folder,
                    "chapter": card.chapter,
                    "source_path": rel,
                    "poll_id": poll_id,
                    "archetype_id": candidate.get("archetype_id"),
                    "opener_id": candidate.get("opener_id"),
                    "question": question,
                    "description": desc,
                    "options": opts,
                    "style_tag": candidate.get("style_tag"),
                    "diversity_ok": bool(candidate.get("diversity_ok", True)),
                    "diversity_reasons": candidate.get("diversity_reasons", []),
                    "diversity_meta": candidate.get("diversity_meta", {}),
                    "critic_scores": scores,
                    "critic_failed": failed_criteria,
                    "critic_notes": {k: notes[k] for k in failed_criteria if notes.get(k)},
                    "critic_overall_pass": bool(critic.get("overall_pass", False)),
                    "critic_plan_version": CRITIC_PLAN_VERSION,
                    "accepted_reason": "critic_pass",
                    "attempt": attempt,
                }
                add_review_entry(module_logs, module_folder, review_entry)

                if candidate.get("archetype_id"):
                    module_archetype_counts.setdefault(module_folder, {})
                    module_archetype_counts[module_folder][candidate["archetype_id"]] = (
                        module_archetype_counts[module_folder].get(candidate["archetype_id"], 0) + 1
                    )
                if candidate.get("opener_id"):
                    module_opener_counts.setdefault(module_folder, {})
                    module_opener_counts[module_folder][candidate["opener_id"]] = (
                        module_opener_counts[module_folder].get(candidate["opener_id"], 0) + 1
                    )
                    global_opener_counts[candidate["opener_id"]] = global_opener_counts.get(candidate["opener_id"], 0) + 1

                # Update registry + mapping
                entry = RegistryEntry(
                    pollId=poll_id,
                    lesson_id=lid,
                    question=question,
                    options=opts,
                    text_fingerprint=stable_hash(cand_text),
                    archetype_id=candidate.get("archetype_id"),
                    opener_id=candidate.get("opener_id"),
                    module=module_folder,
                )
                registry.append(entry)
                mapping[lid] = poll_seg["content"]
                generated += 1
                poll_obj = poll_seg
                log(f"  ✅ Generated poll {poll_id}")
                if args.print_polls:
                    log("\n" + "="*90)
                    log(f"✅ GENERATED POLL for {lid} ({card.title})")
                    log(f"Q: {question}")
                    if desc:
                        log(f"Desc: {desc}")
                    for i,o in enumerate(opts, start=1):
                        log(f"  {i}. {o}")
                    # show critic summary (scores only)
                    try:
                        scores = {k:int(critic.get(k,{}).get("score",0)) for k in ["specificity","distinctness","separability","neutrality","engagement","clarity"]}
                        log(f"Critic scores: {scores}")
                    except Exception:
                        pass
                    log("="*90 + "\n")
                break
            except Exception as e:
                last_err = str(e)
                log(f"  ⚠️ Attempt {attempt} error; retrying. ({last_err})")
                continue

        # If failed, pick best candidate (if any), otherwise record failure
        if poll_obj is None and best_candidate:
            bc = best_candidate
            log(f"  ⚠️ No candidate accepted; selecting best attempt (scores={bc['scores']}, failed={bc['failed_criteria']}, diversity_ok={bc.get('diversity_ok')}).")
            poll_id = f"{lid}__poll__{stable_hash(poll_text(bc['question'], bc['options']))}"
            try:
                poll_seg = build_poll_segment(poll_id, bc["question"], bc["description"], bc["options"])
            except Exception as e:
                # As a last resort, don't crash the whole run; record as a failure.
                failed_lessons += 1
                log(f"  ❌ Best-of-failed candidate could not be validated; lesson failed. ({e})")
                continue
            review_entry = {
                "lesson_id": lid,
                "lesson_title": card.title,
                "module": module_folder,
                "chapter": card.chapter,
                "source_path": rel,
                "poll_id": poll_id,
                "archetype_id": bc.get("archetype_id"),
                "opener_id": bc.get("opener_id"),
                "question": bc["question"],
                "description": bc["description"],
                "options": bc["options"],
                "style_tag": bc.get("style_tag"),
                "diversity_ok": bool(bc.get("diversity_ok", True)),
                "diversity_reasons": bc.get("diversity_reasons", []),
                "diversity_meta": bc.get("diversity_meta", {}),
                "critic_scores": bc["scores"],
                "critic_failed": bc["failed_criteria"],
                "critic_notes": {k: bc["notes"][k] for k in bc["failed_criteria"] if bc["notes"].get(k)},
                "critic_overall_pass": bool(bc["critic"].get("overall_pass", False)),
                "critic_plan_version": CRITIC_PLAN_VERSION,
                "accepted_reason": "best_of_failed",
                "attempt": bc["attempt"],
            }
            add_review_entry(module_logs, module_folder, review_entry)
            if bc.get("archetype_id"):
                module_archetype_counts.setdefault(module_folder, {})
                module_archetype_counts[module_folder][bc["archetype_id"]] = (
                    module_archetype_counts[module_folder].get(bc["archetype_id"], 0) + 1
                )
            if bc.get("opener_id"):
                module_opener_counts.setdefault(module_folder, {})
                module_opener_counts[module_folder][bc["opener_id"]] = (
                    module_opener_counts[module_folder].get(bc["opener_id"], 0) + 1
                )
                global_opener_counts[bc["opener_id"]] = global_opener_counts.get(bc["opener_id"], 0) + 1
            entry = RegistryEntry(
                pollId=poll_id,
                lesson_id=lid,
                question=bc["question"],
                options=bc["options"],
                text_fingerprint=stable_hash(poll_text(bc["question"], bc["options"])),
                archetype_id=bc.get("archetype_id"),
                opener_id=bc.get("opener_id"),
                module=module_folder,
            )
            registry.append(entry)
            mapping[lid] = poll_seg["content"]
            accepted_best += 1
            poll_obj = poll_seg
        elif poll_obj is None:
            failed_lessons += 1
            log(f"  ❌ Failed after {args.max_attempts} attempts")
            if args.debug_failures:
                log(f"  Last error: {last_err}")

    # Write review logs per module
    os.makedirs(review_dir, exist_ok=True)
    for module_name, items in module_logs.items():
        items_sorted = sorted(items, key=lambda x: x.get("lesson_id",""))
        write_json(os.path.join(review_dir, f"{module_name}.json"), items_sorted)

    save_registry(registry_path, registry)
    write_json(review_map_path, mapping)

    log(f"✅ Polls accepted (critic pass): {generated}")
    log(f"⚠️ Polls accepted (best-of-failed): {accepted_best}")
    log(f"⏭️ Skipped (too similar to registry): {skipped_registry}")
    log(f"⏭️ Skipped (lesson already had poll segment): {skipped_existing}")
    log(f"❌ Failed (after retries): {failed_lessons}")
    log(f"✅ Saved registry: {registry_path}")
    log(f"✅ Review export map: {review_map_path}")
    log(f"✅ Review logs written under: {review_dir}")

if __name__ == "__main__":
    main()

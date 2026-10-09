"""ask(question) -> {question_id, answer, sources, forecast_called}  (SPEC section 6)

Flow: understand -> scope check -> forecast tool (only when needed) -> retrieve current
evidence -> compute policy facts -> LLM writes the answer -> validate -> deterministic
fallback if the LLM is unavailable or its answer fails validation.
"""
import hashlib
import re
from datetime import date, timedelta

from . import config, llm
from .forecast_tool import alarm_threshold, forecast_detail, supported_dates
from .policy import facts_for, health_band, next_day, topics_in, valid_rule_sources
from .retrieval import Retriever
from .understand import AREAS, understand

_retriever = None


def _r():
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def _chunks_of(doc_id, n=1):
    return [c for c in _r().chunks if c.doc.doc_id == doc_id][:n]


def _fmt_date(d):
    return f"{date.fromisoformat(d).strftime('%A')} {date.fromisoformat(d).day} {date.fromisoformat(d).strftime('%B %Y')}"


# ---------- deterministic answer (no LLM, or LLM output rejected) ----------

UR = {"hazardous": "khatarnak (hazardous)",
      "unhealthy (not hazardous)": "ghair sehatmand, lekin khatarnak nahi",
      "below 100 (no special precautions for the general public)": "100 se kam (aam logon ke liye khaas ehtiyat zaroori nahi)"}


def _forecast_sentence(fc, ur):
    band, _ = health_band(fc["pm25"])
    src = "team model" if fc["source"] == "team_model" else "organisers' reference forecast"
    if ur:
        return (f"{fc['location']} mein {_fmt_date(fc['target_date'])} ki PM2.5 forecast taqreeban {fc['pm25']:g} µg/m³ "
                f"hai: {UR[band]}. Yeh {src} ki forecast hai; aam ghalti taqreeban ±{config.TYPICAL_ERROR} µg/m³, "
                f"aur achanak ek din ke spike pehle se nahi bataye ja sakte.")
    return (f"Forecast for {fc['location']} on {_fmt_date(fc['target_date'])}: PM2.5 about {fc['pm25']:g} µg/m³, "
            f"which is {band}{_alarm_note(fc, band)}. This is a {src} forecast "
            f"with a typical error of about ±{config.TYPICAL_ERROR} µg/m³; sudden one-day spikes cannot be predicted.")


def _alarm_note(fc, band):
    """Explain an alarm flag that differs from the documents' hazardous band (custom threshold)."""
    if fc["hazardous"] and band != "hazardous":
        return f"; it is flagged by the early-warning hazardous alarm (threshold {alarm_threshold():g} µg/m³)"
    if not fc["hazardous"] and band == "hazardous":
        return f"; the alarm (threshold {alarm_threshold():g} µg/m³) does not flag it"
    return ""


def band_guidance(pm):
    """DOC-01 lines for the band the forecast falls in (DOC-01 lists three bands)."""
    ch = _chunks_of("DOC-01")
    if not ch:
        return []
    text = ch[0].text
    blocks = re.split(r"\n(?=When PM2\.5 is)", "\n" + text)
    key = "165" if pm >= config.HAZARD else ("100" if pm >= config.UNHEALTHY else "below 100")
    for b in blocks:
        head = b.strip().splitlines()[0] if b.strip() else ""
        if (key == "165" and "hazardous range" in head) or (key == "100" and "100–164" in head) \
                or (key == "below 100" and "below 100" in head):
            return [re.sub(r"^[-*\s]+", "", l).strip() for l in b.strip().splitlines()[1:] if l.strip()]
    return []


def _bullets(chunk, n=3):
    lines = [re.sub(r"^[-*>\s]+", "", l).replace("**", "").strip() for l in chunk.text.splitlines()]
    lines = [l for l in lines if l and not l.lower().startswith(("this policy", "this guidance", "this notice"))]
    return lines[:n]


def template_answer(u, plan):
    ur, parts, used = u.roman_urdu, [], []
    for msg in plan["scope_msgs"]:
        parts.append(msg)
    for fc in plan["forecasts_ok"][:1]:
        parts.append(_forecast_sentence(fc, ur))
        g = band_guidance(fc["pm25"])
        if g and plan["want_docs"]:
            parts.append(("DOC-01 hidayat: " if ur else "Health advice for this level (DOC-01): ") + " ".join(
                x.rstrip(".") + "." for x in g))
        used.append("DOC-01")
    if plan.get("city_summary"):
        parts.append(plan["city_summary"] + " (bands: DOC-01)")
        used.append("DOC-01")
    for f, s in plan["facts"][1:]:  # facts[0] restates the forecast band
        parts.append(f"{f} ({s})")
        used.append(s)
    asked_unofficial = re.search(r"communit|bulletin|resident|neighbo", u.question.lower())
    ranked = sorted(plan["evidence"], key=lambda c: (not c.doc.official and not asked_unofficial,
                                                     not c.doc.current and not u.asks_old_policy))
    if u.asks_old_policy:  # show the old rule, then the rule that replaced it
        ranked = sorted(ranked, key=lambda c: (c.doc.current, c.chunk_id.endswith("#0") and c.doc.doc_id == "DOC-03"))
    asks_scope = re.search(r"cover|scope|which areas|cit(y|ies)|sensor|network|aqi|-999|data", u.question.lower())
    for c in ranked:
        if c.doc.doc_id in ("DOC-13", "DOC-06") and not asks_scope and not plan["scope_msgs"]:
            continue  # scope documents only when the question is about scope
        if c.doc.doc_id in used or len(used) >= 3 or c.doc.doc_id in ("DOC-13", "DOC-06") and plan["scope_msgs"]:
            continue
        if not plan["want_docs"]:
            break
        b = _bullets(c)
        if not b:
            continue
        label = f"{'Hidayat' if ur else 'Guidance'} from {c.doc.title} ({c.doc.doc_id})"
        if not c.doc.current:
            label += " [SUPERSEDED, replaced by " + (c.doc.superseded_by or "a newer document") + "]"
        if not c.doc.official:
            label += " [unofficial community source]"
        parts.append(f"{label}: " + " ".join(x.rstrip(".") + "." for x in b))
        used.append(c.doc.doc_id)
    for s in plan["scope_sources"]:
        used.append(s)
    if not parts:
        parts.append("Mujhe diye gaye documents mein is sawal ka jawab nahi mila." if ur else
                     "I could not find this in the supplied documents, so I cannot answer it reliably.")
    return " ".join(parts), list(dict.fromkeys(used))


# ---------- validation of LLM output ----------

NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def _numbers(text):
    text = re.sub(r"DOC-\d+", " ", text.replace(",", ""))  # document IDs are not numeric claims
    return {float(x) for x in NUM.findall(text)}


def validate(out, u, plan):
    """Return (ok, reason). Rejects unknown citations, unsupported numbers, injected claims."""
    ids = {c.doc.doc_id for c in plan["evidence"]}
    cited = [s for s in out["used_source_ids"] if s in ids]
    if len(cited) != len(out["used_source_ids"]):
        return False, f"cited ids not retrieved: {set(out['used_source_ids']) - ids}"
    allowed = set()
    for c in plan["evidence"]:
        allowed |= _numbers(c.text + " " + c.doc.title + " " + c.heading)
    for f, _ in plan["facts"]:
        allowed |= _numbers(f)
    for fc in plan["forecasts_ok"]:
        p = fc["pm25"]
        allowed |= {p, round(p), float(int(p)), round(p, 1)}
        d = date.fromisoformat(fc["target_date"])
        allowed |= {d.day, d.year, d.month}
    for s in plan["scope_msgs"] + [plan.get("city_summary") or ""]:
        allowed |= _numbers(s)
    allowed |= _numbers(u.question) | {config.TYPICAL_ERROR, 2.5, 15, 13, 10, 2026, alarm_threshold()}
    for d in supported_dates():
        dd = date.fromisoformat(d)
        allowed |= {dd.day, dd.month}
    bad = {x for x in _numbers(out["answer"]) if x not in allowed}
    if bad:
        return False, f"unsupported numbers {sorted(bad)}"
    a = out["answer"].lower()
    if plan["forecasts_ok"] and re.search(r"below 50|air (quality )?is (completely )?safe|no (health )?risk", a) \
            and min(f["pm25"] for f in plan["forecasts_ok"]) >= 50:
        return False, "claims safe air against the forecast"
    if "doc-02" in cited and not u.asks_old_policy:
        return False, "used superseded DOC-02"
    if plan["want_docs"] and plan["evidence"] and not cited and not plan["forecasts_ok"]:
        return False, "no sources cited for a document question"
    return True, "ok"


# ---------- main entry ----------

def _qid(question, given):
    if given:
        return str(given)
    return "q-" + hashlib.sha1(question.encode("utf-8")).hexdigest()[:8]


def _plan(u):
    """Decide forecast calls, scope messages and evidence. Returns (plan, forecast_called)."""
    ur = u.roman_urdu
    plan = {"scope_msgs": [], "scope_sources": [], "forecasts_ok": [], "facts": [], "evidence": [],
            "forecast_unavailable": False, "want_docs": u.wants_advice, "city_summary": None}
    called = False
    window = supported_dates()
    areas = ", ".join(AREAS)
    topics = topics_in(u.question)

    if u.loc_kind == "other":
        plan["scope_msgs"].append(
            f"{u.other_place} ke liye koi forecast dastyab nahi. Yeh nizaam sirf Lahore ke 15 ilaqon ka ahata karta hai: {areas}."
            if ur else
            f"No forecast is available for {u.other_place}: this system only covers 15 areas of Lahore ({areas}), "
            f"and the supplied policies apply to Punjab/Lahore only.")
        plan["scope_sources"] += ["DOC-13"]
        plan["evidence"] += _chunks_of("DOC-13")
        plan["want_docs"] = False
    elif u.wants_forecast:
        tgt = u.target
        if tgt is None and u.date_how == "vague":
            plan["scope_msgs"].append(
                f"Please specify a date. Forecasts are available for {window[0]} to {window[-1]}." if not ur else
                f"Barah-e-meharbani tareekh batayen. Forecast {window[0]} se {window[-1]} tak dastyab hai.")
            plan["scope_sources"] += ["DOC-13"]
            plan["evidence"] += _chunks_of("DOC-13")
        else:
            assumed = tgt is None
            tgt = tgt or date.fromisoformat(config.TODAY) + timedelta(days=1)
            d = tgt.isoformat()
            if assumed:
                plan["scope_msgs"].append(f"(No date given; assuming tomorrow, {_fmt_date(d)}.)" if not ur else
                                          f"(Tareekh nahi di gayi; kal, {_fmt_date(d)}, maan rahe hain.)")
            targets = [u.area] if u.area else list(AREAS)
            results = []
            for a in targets:
                fc, reason = forecast_detail(a, d)
                called = True
                results.append((fc, reason))
            oks = [fc for fc, r in results if fc["status"] == "ok"]
            if not oks:
                plan["forecast_unavailable"] = True
                reason = results[0][1]
                where = u.area or "Lahore"
                plan["scope_msgs"].append(
                    f"{where} ke liye {d} ki koi forecast dastyab nahi ({reason}). Forecast sirf {window[0]} se "
                    f"{window[-1]} tak ke liye hai." if ur else
                    f"No forecast is available for {where} on {d} ({reason}). Forecasts cover {window[0]} to "
                    f"{window[-1]} for the 15 covered Lahore areas; I will not estimate a number outside that.")
                plan["scope_sources"] += ["DOC-13"]
                plan["evidence"] += _chunks_of("DOC-13")
            elif u.area:
                fc = oks[0]
                note = _alarm_note(fc, health_band(fc["pm25"])[0])
                if note:
                    plan["alarm_note"] = note.lstrip("; ").capitalize() + "."
                plan["forecasts_ok"] = [fc]
                nd = None
                if "vehicle" in topics:
                    nd, _ = forecast_detail(u.area, next_day(d))
                plan["facts"] = facts_for(fc, topics, nd)
            else:  # "Lahore" or no place: city-wide overview, ask for an area
                hi = max(oks, key=lambda f: f["pm25"])
                lo = min(oks, key=lambda f: f["pm25"])
                haz = [f["location"] for f in oks if f["hazardous"]]
                plan["city_summary"] = (
                    f"Across the 15 covered Lahore areas on {_fmt_date(d)}, forecast PM2.5 ranges from {lo['pm25']:g} "
                    f"({lo['location']}) to {hi['pm25']:g} µg/m³ ({hi['location']}); "
                    + (f"hazardous alarm (≥ {alarm_threshold():g}) for: {', '.join(haz)}." if haz
                       else f"no area triggers the hazardous alarm (≥ {alarm_threshold():g}).")
                    + " Name your area for a specific forecast.")
                plan["forecasts_ok_city"] = oks
                plan["facts"] = [(f"Highest area forecast {hi['pm25']:g} µg/m³ is in the "
                                  f"{health_band(hi['pm25'])[0]} band.", "DOC-01")]

    # drop coded rules whose documents are missing or changed (e.g. judges' own documents)
    valid = valid_rule_sources(_r().docs)
    plan["facts"] = [(f, src) for f, src in plan["facts"] if src in valid]
    # evidence: always include the DOC-01 band guidance when a band is stated
    if (plan["forecasts_ok"] or plan.get("city_summary")) and "DOC-01" in valid:
        plan["evidence"] += _chunks_of("DOC-01", 1)
    for _, s in plan["facts"]:
        if s not in {c.doc.doc_id for c in plan["evidence"]}:
            plan["evidence"] += _chunks_of(s, 2)
    if plan["want_docs"]:
        extra = ""
        if plan["forecasts_ok"]:
            extra = "hazardous 165" if plan["forecasts_ok"][0]["hazardous"] else "unhealthy 100 164 precautions"
        for _, c in _r().search(u.retrieval_query, include_superseded=u.asks_old_policy, extra_terms=extra):
            if c.chunk_id not in {x.chunk_id for x in plan["evidence"]}:
                plan["evidence"].append(c)
    return plan, called


def _ask(question, question_id=None, client=None):
    if isinstance(question, dict):
        question_id = question_id or question.get("question_id") or question.get("id")
        question = question.get("question") or question.get("text") or ""
    question = str(question).strip()
    u = understand(question)
    plan, called = _plan(u)
    trace = {"understanding": u, "plan": plan, "llm": None, "validation": None}

    answer, sources = None, None
    if llm.available() or client is not None:
        try:
            fc_payload = plan["forecasts_ok"][0] if plan["forecasts_ok"] else (
                {"status": "unavailable"} if plan["forecast_unavailable"] else None)
            facts = list(plan["facts"]) + [(m, "scope") for m in plan["scope_msgs"]]
            if plan.get("city_summary"):
                facts.append((plan["city_summary"], "forecast"))
            if plan.get("alarm_note"):
                facts.append((plan["alarm_note"], "forecast alarm setting"))
            out = llm.write_answer(question, "Roman Urdu" if u.roman_urdu else "English",
                                   plan["evidence"], fc_payload, facts, client=client)
            ok, why = validate(out, u, plan)
            trace["llm"], trace["validation"] = out, why
            if ok:
                answer, sources = out["answer"], list(dict.fromkeys(out["used_source_ids"]))
        except llm.LLMUnavailable as e:
            trace["validation"] = f"llm unavailable: {e}"
    if answer is None:
        answer, sources = template_answer(u, plan)
        valid_ids = {c.doc.doc_id for c in plan["evidence"]}
        sources = [s for s in sources if s in valid_ids]
        trace["mode"] = "deterministic"
    else:
        trace["mode"] = "llm"
    result = {"question_id": _qid(question, question_id), "answer": answer, "sources": sources,
              "forecast_called": called}
    return result, trace


def ask(question, question_id=None):
    """SPEC contract. `question` may be a string or a dict with "question" and "question_id"/"id"."""
    return _ask(question, question_id)[0]

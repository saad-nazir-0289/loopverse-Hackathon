"""Turn a forecast into the thresholds the CURRENT documents define, each tied to its source.

These facts are computed in code, so the LLM never has to do threshold arithmetic and
cannot be talked into a different conclusion by retrieved text.
"""
from datetime import date, timedelta

from .config import HAZARD, SCHOOL_FULL_CLOSURE, UNHEALTHY


# Each coded rule is only used if its document exists and still states these thresholds.
RULE_REQUIRES = {"DOC-01": ["165", "100"], "DOC-03": ["165", "250", "half-day"],
                 "DOC-05": ["165", "consecutive"], "DOC-08": ["165", "kiln"]}


def valid_rule_sources(docs):
    by_id = {d.doc_id: d.text for d in docs if d.current}
    return {i for i, needs in RULE_REQUIRES.items() if i in by_id and all(n in by_id[i] for n in needs)}


def health_band(pm):
    if pm >= HAZARD:
        return "hazardous", "DOC-01"
    if pm >= UNHEALTHY:
        return "unhealthy (not hazardous)", "DOC-01"
    return "below 100 (no special precautions for the general public)", "DOC-01"


def facts_for(fc, topics, next_day_fc=None):
    """List of (fact, doc_id) for an ok forecast. topics: set of words found in the question."""
    pm, area, d = fc["pm25"], fc["location"], fc["target_date"]
    band, src = health_band(pm)
    facts = [(f"Forecast PM2.5 {pm:g} µg/m³ for {area} on {d} falls in the {band} band.", src)]
    if "school" in topics:
        if pm >= SCHOOL_FULL_CLOSURE:
            facts.append(("Under the current school policy schools close fully (forecast ≥ 250).", "DOC-03"))
        elif pm >= HAZARD:
            facts.append(("Under the current school policy schools switch to half-day timing, closing by 12:00 PM "
                          "(forecast 165–249), and outdoor sports, assemblies and recess are suspended.", "DOC-03"))
        else:
            facts.append(("The forecast is below 165, so the current school policy's half-day and closure triggers "
                          "are not met; schools follow normal timing.", "DOC-03"))
        facts.append(("Decisions are made the evening before using the next-day forecast for the school's area; "
                      "parents are notified by SMS by 7:00 PM.", "DOC-03"))
    if "vehicle" in topics:
        two_days = next_day_fc is not None and next_day_fc.get("status") == "ok"
        if pm >= HAZARD and two_days and next_day_fc["pm25"] >= HAZARD:
            facts.append(("Forecast is hazardous on two consecutive days, which meets the odd-even trigger.", "DOC-05"))
        elif pm >= HAZARD:
            facts.append(("Odd-even applies only after two or more consecutive hazardous forecast days; "
                          "this day alone does not establish that.", "DOC-05"))
        else:
            facts.append(("The forecast is below 165, so the odd-even restriction trigger is not met.", "DOC-05"))
    if "construction" in topics:
        facts.append((("Forecast is hazardous, so construction dust work halts 7 AM–7 PM, open burning is banned "
                       "and non-compliant brick kilns suspend operation.") if pm >= HAZARD else
                      "The forecast is below 165, so the hazardous-day construction and kiln restrictions are not triggered.",
                      "DOC-08"))
    return facts


def topics_in(text):
    t = text.lower()
    out = set()
    if any(w in t for w in ("school", "sakool", "skool", "class", "recess", "assembl", "student", "madrasa")):
        out.add("school")
    if any(w in t for w in ("odd", "even", "vehicle", " car", "plate", "drive", "gari", "gaari", "gaadi")):
        out.add("vehicle")
    if any(w in t for w in ("construction", "kiln", "bhatta", "burning", "factory", "industr", "tameer")):
        out.add("construction")
    return out


def next_day(d):
    return (date.fromisoformat(d) + timedelta(days=1)).isoformat()

"""Deterministic question understanding: language, location, target date, and what is being asked.

Rule-based on purpose: routing decides whether the forecast tool runs, so it must be
predictable and testable (forecast_called has to be truthful).
"""
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from .config import METADATA, TODAY

# ---------- Roman Urdu ----------
# Common Roman Urdu spellings -> English retrieval terms
ROMAN_URDU = {
    "kal": "tomorrow", "parson": "day after tomorrow", "aaj": "today", "aj": "today",
    "hawa": "air", "fiza": "air", "aloodgi": "pollution", "aludgi": "pollution", "dhuan": "smog",
    "dhund": "smog", "sakool": "school", "skool": "school", "schools": "school", "madrasa": "school",
    "bachay": "children", "bachey": "children", "bachon": "children", "bache": "children", "bacho": "children",
    "bacha": "child", "band": "close closed", "khule": "open", "khula": "open", "chutti": "closure holiday",
    "chhutti": "closure holiday", "aadha": "half", "adha": "half", "maask": "mask", "naqab": "mask",
    "bahar": "outdoor outside", "baahar": "outdoor outside", "ghar": "home indoor", "andar": "indoor",
    "saans": "breathing breath", "khansi": "cough", "dama": "asthma", "buzurg": "elderly", "buzurgon": "elderly", "ehtiyat": "precautions", "ehtiyaat": "precautions", "boorhay": "elderly",
    "hamla": "pregnant", "hamila": "pregnant", "gari": "vehicle car", "gaari": "vehicle car", "gaadi": "vehicle car",
    "gaariyan": "vehicles", "haspatal": "hospital", "aspatal": "hospital", "mehfooz": "safe", "mahfooz": "safe",
    "khatarnak": "hazardous dangerous", "khatarnaak": "hazardous dangerous", "warzish": "exercise",
    "daurna": "running exercise", "khel": "sports play", "bhatta": "brick kiln", "bhattay": "brick kilns",
    "tameer": "construction", "kiraya": "fare", "dawai": "medication medicine", "ilaaj": "treatment",
    "safai": "purifier", "tez": "high", "zyada": "high more", "ziyada": "high more", "kam": "low less",
    "barish": "rain", "sardi": "cold", "garmi": "heat", "kitni": "how much", "kitna": "how much",
    "kaisi": "how", "kaisa": "how", "kab": "when", "kahan": "where", "kyun": "why", "kyon": "why",
    "pabandi": "restriction ban", "qanoon": "rule policy", "hukam": "order directive",
}
ROMAN_MARKERS = {"kya", "kia", "hai", "hain", "hy", "ka", "ki", "ke", "ko", "se", "mein", "main", "hoga", "hogi",
                 "honge", "nahi", "nahin", "chahiye", "chaiye", "karna", "karen", "karein", "kal", "aaj", "bachon",
                 "bachay", "kitni", "kitna", "kaisi", "kaisa", "kab", "kyun", "aur", "ya", "lekin", "agar", "bhi",
                 "hawa", "par", "pe", "wala", "wali", "sakte", "sakta", "sakti", "jana", "jaye", "jayen", "rahe"}


def is_roman_urdu(q):
    words = re.findall(r"[a-z]+", q.lower())
    return sum(w in ROMAN_MARKERS for w in words) >= 2


def normalize_for_retrieval(q):
    words = re.findall(r"[a-z0-9.]+", q.lower())
    return " ".join(f"{w} {ROMAN_URDU[w]}" if w in ROMAN_URDU else w for w in words)


# ---------- locations ----------
_meta = pd.read_csv(METADATA)
AREAS = dict(zip(_meta.area, _meta.sensor_id))  # "Gulberg" -> "S01"
SENSOR_TO_AREA = {s: a for a, s in AREAS.items()}
ALIASES = {
    "defence": "DHA", "defense": "DHA", "d.h.a": "DHA", "dha": "DHA",
    "johar town": "Johar Town", "johar": "Johar Town", "model town": "Model Town",
    "thokar": "Thokar Niaz Baig", "thokar niaz baig": "Thokar Niaz Baig", "niaz baig": "Thokar Niaz Baig",
    "allama iqbal town": "Allama Iqbal Town", "iqbal town": "Allama Iqbal Town", "a.i.t": "Allama Iqbal Town",
    "cantt": "Cantt", "cantonment": "Cantt", "faisal town": "Faisal Town", "township": "Township",
    "walled city": "Walled City", "old city": "Walled City", "andrun shehr": "Walled City", "androon shehr": "Walled City",
    "raiwind road": "Raiwind Road", "raiwind": "Raiwind Road", "bahria town": "Bahria Town", "bahria": "Bahria Town",
    "airport road": "Airport Road", "gulberg": "Gulberg", "wagah": "Wagah", "wahga": "Wagah", "shadman": "Shadman",
}
# Places that are clearly NOT covered (other cities, and Lahore areas outside the network)
OTHER_PLACES = ["karachi", "islamabad", "rawalpindi", "quetta", "peshawar", "multan", "faisalabad", "sialkot",
                "gujranwala", "hyderabad", "sukkur", "bahawalpur", "sargodha", "sheikhupura", "kasur", "delhi",
                "new delhi", "mumbai", "amritsar", "dubai", "london", "new york", "beijing", "kabul",
                "garden town", "valencia", "wapda town", "gulshan", "samanabad", "ichhra", "mughalpura",
                "shahdara", "lake city", "askari", "punjab university"]


def find_location(q):
    """Return (kind, value): ("area", "Gulberg") / ("other", "Karachi") / ("lahore", None) / (None, None)."""
    ql = " " + re.sub(r"[^a-z0-9. ]", " ", q.lower()) + " "
    m = re.search(r"\bs(0?[1-9]|1[0-5])\b", ql)
    if m:
        return "area", SENSOR_TO_AREA[f"S{int(m.group(1)):02d}"]
    for alias in sorted(ALIASES, key=len, reverse=True):  # longest first: "allama iqbal town" before "iqbal town"
        if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", ql):
            return "area", ALIASES[alias]
    for p in OTHER_PLACES:
        if re.search(rf"(?<![a-z]){re.escape(p)}(?![a-z])", ql):
            return "other", p.title()
    if re.search(r"\blahore\b", ql):
        return "lahore", None
    return None, None


# ---------- dates ----------
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
WEEKDAYS_UR = {"peer": 0, "pir": 0, "mangal": 1, "budh": 2, "jumerat": 3, "jumeraat": 3, "juma": 4, "jumma": 4,
               "hafta": 5, "haftay": 5, "itwar": 6, "itwaar": 6}


def find_date(q, today=None):
    """Return (date or None, how it was found). Relative words resolve against the scenario clock."""
    today = today or date.fromisoformat(TODAY)
    ql = q.lower()
    m = re.search(r"\b(20\d\d)-(\d{1,2})-(\d{1,2})\b", ql)
    if m:
        return _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3))), "iso"
    m = re.search(r"\b(\d{1,2})[/.](\d{1,2})[/.](20\d\d)\b", ql)  # Pakistani day/month/year
    if m:
        return _safe_date(int(m.group(3)), int(m.group(2)), int(m.group(1))), "dmy"
    mon = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
    m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{mon}\.?(?:,?\s+(20\d\d))?", ql) or None
    if m:
        return _safe_date(int(m.group(3) or today.year), MONTHS[m.group(2)[:3]], int(m.group(1))), "day month"
    m = re.search(rf"\b{mon}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(20\d\d))?", ql)
    if m:
        return _safe_date(int(m.group(3) or today.year), MONTHS[m.group(1)[:3]], int(m.group(2))), "month day"
    if re.search(r"day after tomorrow|\bparson\b|\bparsoon\b", ql):
        return today + timedelta(days=2), "day after tomorrow"
    if re.search(r"\btomorrow\b|\bkal\b|\btmrw\b|\bnext day\b", ql):
        return today + timedelta(days=1), "tomorrow"
    if re.search(r"\btoday\b|\baaj\b|\btonight\b|\bright now\b|\bcurrently\b", ql):
        return today, "today"
    if re.search(r"\byesterday\b", ql):
        return today - timedelta(days=1), "yesterday"
    m = re.search(r"\bin (\d{1,2}) days\b", ql)
    if m:
        return today + timedelta(days=int(m.group(1))), "in n days"
    for i, wd in enumerate(WEEKDAYS):
        if re.search(rf"\b{wd}\b", ql):
            return today + timedelta(days=(i - today.weekday() - 1) % 7 + 1), f"next {wd}"
    for w, i in WEEKDAYS_UR.items():
        if re.search(rf"\b{w}\b", ql):
            return today + timedelta(days=(i - today.weekday() - 1) % 7 + 1), f"next {w}"
    if re.search(r"next week|this week|weekend|next month|last week|last year", ql):
        return None, "vague"
    return None, None


def _safe_date(y, mo, d):
    try:
        return date(y, mo, d)
    except ValueError:
        return None


# ---------- intent ----------
FORECAST_WORDS = r"forecast|predict|pm2\.?5|pm 2\.?5|air quality|aqi|how bad|how polluted|pollution level|smog level|" \
                 r"hazardous|will it be|level|reading|kitni|kitna|kaisi|kaisa|safe|mehfooz|mahfooz|khatarnak"
ADVICE_WORDS = r"should|can i|can we|could|advice|recommend|safe|precaution|mask|exercise|jog|run|walk|outdoor|" \
               r"outside|children|kids|school|elderly|pregnan|asthma|heart|purifier|window|hospital|close|closure|" \
               r"half-day|half day|policy|rule|regulation|odd|even|vehicle|car|construction|kiln|burning|metro|" \
               r"transport|bus|sports|recess|assembly|what to do|kya karna|chahiye|bachon|bachay|sakool|band|" \
               r"maask|bahar|warzish|dama|buzurg|gari|haspatal|pabandi"
OLD_POLICY = r"\b(old|previous|earlier|former|2023|2025|version 1|v1|superseded|used to|before the new|purani|pehle)\b"


@dataclass
class Understanding:
    question: str
    roman_urdu: bool
    loc_kind: str | None
    area: str | None
    other_place: str | None
    target: date | None
    date_how: str | None
    wants_forecast: bool
    wants_advice: bool
    asks_old_policy: bool
    retrieval_query: str
    notes: list = field(default_factory=list)


def understand(q):
    kind, val = find_location(q)
    tgt, how = find_date(q)
    ql = q.lower()
    wants_forecast = bool(kind in ("area", "lahore") and (tgt or re.search(FORECAST_WORDS, ql))) \
        or bool(tgt and re.search(FORECAST_WORDS, ql))
    wants_advice = bool(re.search(ADVICE_WORDS, ql)) or not wants_forecast
    return Understanding(
        question=q, roman_urdu=is_roman_urdu(q), loc_kind=kind,
        area=val if kind == "area" else None, other_place=val if kind == "other" else None,
        target=tgt, date_how=how, wants_forecast=wants_forecast, wants_advice=wants_advice,
        asks_old_policy=bool(re.search(OLD_POLICY, ql)), retrieval_query=normalize_for_retrieval(q))

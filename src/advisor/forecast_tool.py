"""forecast(location, target_date): the SPEC section 5 contract.

Validates the location against the 15 covered areas (or sensor IDs) and the date
against the supported forecast window. Never returns a guessed number: any invalid
input or failure gives status "unavailable" with pm25 = None.
"""
from datetime import date

import pandas as pd

from . import config
from .understand import ALIASES, AREAS, SENSOR_TO_AREA

_cache = {}


def _load():
    """Team predictions by default; the organisers' reference forecast only as a fallback."""
    if "table" in _cache:
        return _cache["table"], _cache["source"]
    use_ref = config.FORECAST_SOURCE == "reference" or not config.PREDICTIONS.exists()
    path, source = (config.REFERENCE_FORECAST, "reference_fallback") if use_ref else (config.PREDICTIONS, "team_model")
    t = pd.read_csv(path)
    if "hazardous" not in t.columns:
        t["hazardous"] = (t["predicted_pm25"] >= config.HAZARD).astype(int)
    t["target_date"] = pd.to_datetime(t["target_date"]).dt.strftime("%Y-%m-%d")
    t = t.set_index(["sensor_id", "target_date"])
    _cache.update(table=t, source=source)
    return t, source


def supported_dates():
    h = pd.read_csv(config.HOLDOUT)
    return sorted(h["target_date"].unique())


def resolve_location(location):
    """Area name, alias or sensor ID -> canonical area name, or None."""
    if not isinstance(location, str):
        return None
    s = location.strip()
    if s.upper() in SENSOR_TO_AREA:
        return SENSOR_TO_AREA[s.upper()]
    for area in AREAS:
        if s.lower() == area.lower():
            return area
    return ALIASES.get(s.lower())


def forecast_detail(location, target_date):
    """Same as forecast() plus a machine-readable reason, for the assistant's own use."""
    area = resolve_location(location)
    out = {"location": area or (location if isinstance(location, str) else str(location)),
           "target_date": str(target_date), "pm25": None, "hazardous": None, "status": "unavailable",
           "source": "team_model"}
    try:
        t, source = _load()
        out["source"] = source
    except Exception as e:  # missing or unreadable forecast file: say so, never guess
        return out, f"forecast data unavailable ({type(e).__name__})"
    if area is None:
        return out, "location is not one of the 15 covered Lahore areas"
    try:
        d = date.fromisoformat(str(target_date)).isoformat()
    except ValueError:
        return out, "target_date is not a valid YYYY-MM-DD date"
    out["target_date"] = d
    window = supported_dates()
    if d not in window:
        return out, f"date outside the supported forecast window {window[0]} to {window[-1]}"
    key = (AREAS[area], d)
    if key not in t.index:
        return out, "no forecast row for this area and date"
    row = t.loc[key]
    pm = float(row["predicted_pm25"])
    if not pd.notna(pm):
        return out, "forecast value missing"
    out.update(pm25=round(pm, 1), hazardous=bool(int(row["hazardous"])), status="ok")
    return out, "ok"


def forecast(location, target_date):
    """SPEC contract: {location, target_date, pm25, hazardous, status, source}."""
    return forecast_detail(location, target_date)[0]

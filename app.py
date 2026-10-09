"""Simple UI for the Lahore Smog Intelligence system.

    streamlit run app.py
"""
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from advisor import config, llm  # noqa: E402
from advisor.ask import _ask, _r  # noqa: E402
from advisor.forecast_tool import forecast, supported_dates  # noqa: E402
from advisor.understand import AREAS  # noqa: E402

st.set_page_config(page_title="Lahore Smog Intelligence", page_icon="🌫️", layout="wide")
SPEC_THRESHOLD = config.HAZARD  # 165, SPEC section 3


@st.cache_data
def predictions():
    p = pd.read_csv(config.PREDICTIONS)
    meta = pd.read_csv(config.METADATA)[["sensor_id", "area"]]
    return p.merge(meta, on="sensor_id")


@st.cache_data
def validation():
    """Walk-forward predictions of the submitted model, to show what a threshold would have done."""
    try:
        scores = pd.read_csv(ROOT / "outputs" / "walk_forward_scores.csv", index_col=0)
        model = scores.drop(index=["persistence", "mean7"], errors="ignore").index[0]
        cv = pd.read_csv(ROOT / "outputs" / "walk_forward_predictions.csv")
        return model, cv[["y", model]].rename(columns={model: "pred"})
    except Exception:
        return None, None


st.title("Lahore Smog Intelligence")
st.caption(f"Next-day PM2.5 forecasts for 15 Lahore areas ({supported_dates()[0]} to {supported_dates()[-1]}) "
           "and an advisory assistant that cites the supplied guidance.")

with st.sidebar:
    st.header("Hazardous alarm")
    thr = st.number_input("Alarm threshold (µg/m³)", min_value=50.0, max_value=400.0, value=SPEC_THRESHOLD,
                          step=5.0, help="A forecast at or above this value is flagged hazardous. SPEC default: 165.")
    config.ALARM_THRESHOLD = float(thr)  # used by forecast(), ask(), charts and the download below
    if thr != SPEC_THRESHOLD:
        st.info(f"Custom alarm: {thr:g} µg/m³ (SPEC uses {SPEC_THRESHOLD:g}). Document health bands and "
                "policy rules still use the documents' own thresholds.")
    model, cv = validation()
    if cv is not None:
        y, flag = cv.y >= SPEC_THRESHOLD, cv.pred >= thr
        caught, fa = int((y & flag).sum()), int((~y & flag).sum())
        st.caption(f"Validation ({len(cv)} past forecasts, {int(y.sum())} truly hazardous) at this threshold: "
                   f"**{caught} caught**, **{fa} false alarms**, {int(flag.sum())} flagged.")
    st.divider()
    st.header("System")
    st.write(f"**Answer writer:** {'LLM: ' + config.LLM_MODEL if llm.available() else 'deterministic (no LLM)'}")
    st.write(f"**Scenario date (today):** {config.TODAY}")
    st.write(f"**Forecast source:** {'team model' if config.PREDICTIONS.exists() else 'reference fallback'}")
    st.caption("Documents are treated as evidence, never as instructions. Out-of-scope places and dates "
               "get no number.")

tab_ask, tab_fc, tab_pred = st.tabs(["Ask the assistant", "Forecast", "All predictions"])

with tab_ask:
    examples = ["Should schools in Walled City close on 5 November?",
                "Is it safe for children to play outside in Gulberg tomorrow?",
                "Will Lahore be hazardous tomorrow?",
                "What mask protects against PM2.5?",
                "What is the PM2.5 forecast for Karachi tomorrow?",
                "Kal Johar Town mein bachon ko school bhejna chahiye?"]
    pick = st.selectbox("Try an example, or type your own below", ["(type your own)"] + examples)
    q = st.text_area("Question", value="" if pick == "(type your own)" else pick, height=80)
    if st.button("Ask", type="primary", disabled=not q.strip()):
        with st.spinner("Retrieving evidence and checking the forecast..."):
            res, trace = _ask(q)
        st.markdown(res["answer"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Forecast called", "yes" if res["forecast_called"] else "no")
        c2.metric("Sources", ", ".join(res["sources"]) or "none")
        c3.metric("Answer path", trace.get("mode", "?"))
        if res["sources"]:
            with st.expander("Cited documents"):
                docs = {d.doc_id: d for d in _r().docs}
                for s in res["sources"]:
                    d = docs[s]
                    st.markdown(f"**{s}: {d.title}** · {d.authority} · {d.published_date} · status: {d.status}")
                    st.text(d.text[:1500])
        with st.expander("Raw output (SPEC schema)"):
            st.json(res)

p = predictions().copy()
p["hazardous"] = (p["predicted_pm25"] >= thr).astype(int)

with tab_fc:
    c1, c2 = st.columns(2)
    area = c1.selectbox("Area", list(AREAS))
    day = c2.selectbox("Target date", supported_dates())
    fc = forecast(area, day)
    if fc["status"] == "ok":
        m1, m2, m3 = st.columns(3)
        m1.metric("Forecast PM2.5 (µg/m³)", f"{fc['pm25']:g}")
        m2.metric(f"Hazardous (≥ {thr:g})", "YES" if fc["hazardous"] else "no")
        m3.metric("Source", fc["source"])
    else:
        st.warning("No forecast available for this area and date.")
    st.json(fc)
    day_df = p[p.target_date == day].sort_values("predicted_pm25", ascending=False)
    bars = alt.Chart(day_df).mark_bar().encode(
        x=alt.X("predicted_pm25:Q", title="Forecast PM2.5 (µg/m³)"),
        y=alt.Y("area:N", sort="-x", title=None),
        color=alt.condition(alt.datum.predicted_pm25 >= thr, alt.value("#d62828"), alt.value("#4c78a8")),
        opacity=alt.condition(alt.datum.area == area, alt.value(1.0), alt.value(0.6)),
        tooltip=["area", "sensor_id", "predicted_pm25", "hazardous"])
    rule = alt.Chart(pd.DataFrame({"x": [thr]})).mark_rule(strokeDash=[4, 4], color="black").encode(x="x")
    st.subheader(f"All areas on {day}")
    st.altair_chart(bars + rule, width="stretch")
    st.caption(f"Dashed line and red bars: alarm threshold {thr:g} µg/m³. Forecasts are 1-10 days ahead of the "
               f"last observation; typical error is about ±{config.TYPICAL_ERROR} µg/m³ and sudden one-day spikes "
               "cannot be predicted.")

with tab_pred:
    flagged = int(p.hazardous.sum())
    st.write(f"**{flagged} of {len(p)}** rows flagged hazardous at the alarm threshold **{thr:g} µg/m³**.")
    st.dataframe(p[["sensor_id", "area", "target_date", "predicted_pm25", "hazardous"]], width="stretch",
                 hide_index=True)
    sub = p[["sensor_id", "target_date", "predicted_pm25", "hazardous"]]  # SPEC section 4 column order
    st.download_button(f"Download predictions.csv (alarm at {thr:g})", sub.to_csv(index=False).encode(),
                       "predictions.csv", "text/csv")
    st.caption("The predictions.csv in the repository is written by `python src/forecast.py` "
               f"(SPEC threshold {SPEC_THRESHOLD:g}). To write it with another threshold: "
               f"`python src/forecast.py --alarm-threshold {thr:g}`.")
    trend = p.groupby("target_date").predicted_pm25.agg(["min", "mean", "max"])
    st.subheader("City-wide forecast by day")
    st.line_chart(trend)

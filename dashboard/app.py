"""Gatekeeper dashboard: live decisions, replay controls, and a form to test a payment."""
from __future__ import annotations

import os
from datetime import datetime

import altair as alt
import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
SIM_URL = os.environ.get("SIM_URL", "http://localhost:8001")
REFRESH_SECONDS = float(os.environ.get("REFRESH_SECONDS", "2"))

# Validated two-slot palette (light surface): allow = blue, suspicious = orange
COLORS = {"allow": "#2a78d6", "suspicious": "#eb6834"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

st.set_page_config(page_title="Gatekeeper", page_icon="🛡️", layout="wide")


@st.cache_resource
def client() -> httpx.Client:
    return httpx.Client(timeout=5)


def get(url: str, **params):
    try:
        r = client().get(url, params=params)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def post(url: str, payload: dict | None = None):
    try:
        r = client().post(url, json=payload)
        if r.status_code >= 400:
            return None, r.json().get("detail", r.text)
        return r.json(), None
    except Exception as e:
        return None, str(e)


def money(x) -> str:
    return "—" if x is None else f"${x:,.0f}"


def reasons_text(v) -> str:
    """Stored reasons (a list of dicts) -> one readable line."""
    if not isinstance(v, list) or not v:
        return ""
    return " · ".join(f"{'▲' if r.get('impact', 0) > 0 else '▼'} {r['name']} = {r['value']}" for r in v)


def pct(x, digits=1) -> str:
    return "—" if x is None else f"{x * 100:.{digits}f}%"


# ---------------------------------------------------------------------------------------------
st.title("🛡️ Gatekeeper")
st.caption("Real-time fraud scoring. Every payment gets **allow** or **suspicious** in milliseconds, with the top reasons for every flagged payment.")

model, model_err = get(f"{API_URL}/model")
if model_err:
    st.error(f"Scoring API unreachable at {API_URL}: {model_err}")
    st.stop()

# ---- Sidebar: replay controls --------------------------------------------------------------
with st.sidebar:
    st.header("Replay the held-out month")
    st.caption("Streams real payments the model has never seen, in their original order.")
    rate = st.slider("Payments per second", 1, 200, 20)
    limit = st.number_input("How many payments", min_value=10, max_value=85_430, value=2_000, step=500)
    c1, c2 = st.columns(2)
    if c1.button("▶ Start", use_container_width=True, type="primary"):
        _, err = post(f"{SIM_URL}/start", {"rate": rate, "limit": int(limit)})
        st.toast(err or f"Replay started at {rate}/s")
    if c2.button("■ Stop", use_container_width=True):
        _, err = post(f"{SIM_URL}/stop")
        st.toast(err or "Replay stopped")
    if st.button("Clear card history", use_container_width=True,
                 help="Forget every card's recent activity before a fresh replay"):
        res, err = post(f"{SIM_URL}/reset-history")
        st.toast(err or f"Cleared {res['deleted_keys']:,} history entries")

    st.divider()
    window = st.selectbox("Show results for", ["Last 5 minutes", "Last 15 minutes", "Last hour", "All time"], index=3)
    minutes = {"Last 5 minutes": 5, "Last 15 minutes": 15, "Last hour": 60, "All time": None}[window]

    st.divider()
    st.subheader("Champion model")
    st.write(f"**{model['model']}** · {model['trees']} trees · {model['features']} features")
    st.write(f"Flags a payment when its fraud score ≥ **{model['threshold']:.3f}**")
    st.caption(model.get("threshold_policy") or "")


# ---- Live area (refreshes on its own) ------------------------------------------------------
@st.fragment(run_every=REFRESH_SECONDS)
def live():
    sim, sim_err = get(f"{SIM_URL}/status")
    if sim and sim["running"]:
        st.progress(sim["progress"], text=f"Replaying: {sim['sent']:,} of {sim['total']:,} payments · "
                                          f"{sim['rate_actual']:.0f}/s · payment time {(sim['payment_time'] or '')[:16].replace('T', ' ')}")
    elif sim_err:
        st.caption(f"Simulator unreachable at {SIM_URL}")

    stats, err = get(f"{API_URL}/stats", **({"minutes": minutes} if minutes else {}))
    if err:
        st.warning(f"Decision log unavailable: {err}")
        return

    tiles = [
        ("Payments scored", f"{stats['decisions']:,}", ""),
        ("Flagged as suspicious", f"{stats['flagged']:,}", f"{pct(stats['flag_rate'])} of payments"),
        ("Fraud caught", pct(stats["catch_rate"]),
         f"{stats['frauds_caught']:,} of {stats['frauds']:,} frauds" if stats["frauds"] else "needs known outcomes"),
        ("False alarms", pct(stats["false_alarm_rate"], 2),
         f"{stats['false_alarms']:,} genuine customers flagged" if stats["genuine"] else "needs known outcomes"),
        ("Fraud stopped", money(stats["fraud_stopped_usd"]),
         f"{money(stats['fraud_missed_usd'])} missed" if stats["frauds"] else ""),
        ("Decision time", f"{(stats['avg_latency_ms'] or 0):.2f} ms",
         f"p99 {(stats['p99_latency_ms'] or 0):.2f} ms · target 50 ms"),
    ]
    for col, (label, value, note) in zip(st.columns(6), tiles):
        col.metric(label, value)
        col.caption(note or " ")

    recent, err = get(f"{API_URL}/decisions/recent", limit=400)
    if err or not recent:
        st.info("No decisions yet. Start a replay from the sidebar, or test a payment below.")
        return
    df = pd.DataFrame(recent)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["truth"] = df["label"].map({1: "fraud", 0: "genuine"}).fillna("unknown")

    st.subheader("Fraud score of the latest payments")
    base = alt.Chart(df).encode(x=alt.X("created_at:T", title=None, axis=alt.Axis(format="%H:%M:%S")))
    dots = base.mark_circle(size=64, opacity=0.85, stroke="#fcfcfb", strokeWidth=1).encode(
        y=alt.Y("fraud_score:Q", title="fraud score", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("decision:N", scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("transaction_id:N", title="transaction"),
                 alt.Tooltip("amount:Q", title="amount ($)", format=",.2f"),
                 alt.Tooltip("fraud_score:Q", title="score", format=".3f"),
                 alt.Tooltip("decision:N"), alt.Tooltip("truth:N"),
                 alt.Tooltip("card_tx_last_hour:Q", title="card payments, last hour"),
                 alt.Tooltip("amount_vs_card_avg:Q", title="× card's usual amount")],
    )
    rule = alt.Chart(pd.DataFrame({"t": [model["threshold"]]})).mark_rule(
        strokeDash=[4, 4], color=INK_2).encode(y="t:Q")
    label = alt.Chart(pd.DataFrame({"t": [model["threshold"]], "x": [df["created_at"].min()]})).mark_text(
        align="left", dy=-6, color=INK_2, fontSize=11).encode(
        x="x:T", y="t:Q", text=alt.value(f"threshold {model['threshold']:.3f}"))
    chart = (dots + rule + label).properties(height=280).configure_axis(
        gridColor=GRID, domainColor=GRID, labelColor=INK_2, titleColor=INK_2
    ).configure_view(strokeWidth=0).configure_legend(labelColor=INK)
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Latest flagged payments, and why")
    flagged = df[df["decision"] == "suspicious"].head(10)
    if flagged.empty:
        st.caption("None in the latest payments.")
    else:
        show = pd.DataFrame({
            "time": flagged["created_at"].dt.strftime("%H:%M:%S"),
            "amount": flagged["amount"].map(lambda v: "—" if pd.isna(v) else f"${v:,.2f}"),
            "score": flagged["fraud_score"].map(lambda v: f"{v:.3f}"),
            "truth": flagged["truth"],
            "card, last hr": flagged["card_tx_last_hour"],
            "top reasons": flagged["reasons"].map(reasons_text) if "reasons" in flagged else "",
        })
        st.dataframe(show, hide_index=True, use_container_width=True,
                     column_config={"top reasons": st.column_config.TextColumn(width="large")})
        st.caption("Top reasons: the three features that pushed the fraud score up (▲) the most, "
                   "from SHAP values. C, D, M, V and id_ fields are anonymised by the data provider.")

    with st.expander("All latest decisions (table)"):
        t = df.head(200)

        def outcome(r):
            if r["truth"] == "unknown":
                return ""
            if r["truth"] == "fraud":
                return "✅ fraud caught" if r["decision"] == "suspicious" else "❌ fraud missed"
            return "⚠️ false alarm" if r["decision"] == "suspicious" else "✓ genuine allowed"

        def num(v, fmt):
            return "" if v is None or pd.isna(v) else fmt.format(v)

        table = pd.DataFrame({
            "time": t["created_at"].dt.strftime("%H:%M:%S"),
            "transaction": t["transaction_id"],
            "amount": t["amount"].map(lambda v: num(v, "${:,.2f}")),
            "product": t["product"].fillna(""),
            "fraud score": t["fraud_score"].map(lambda v: f"{v:.3f}"),
            "decision": t["decision"],
            "outcome": t.apply(outcome, axis=1),
            "safety rule": t["rule"].fillna("") if "rule" in t else "",
            "top reasons": t["reasons"].map(reasons_text) if "reasons" in t else "",
            "card payments, last hour": t["card_tx_last_hour"].map(lambda v: num(v, "{:.0f}")),
            "card's usual amount": t["card_avg_amount"].map(lambda v: num(v, "${:,.2f}")),
            "× usual": t["amount_vs_card_avg"].map(lambda v: num(v, "{:.1f}×")),
            "new device": t["new_device"].map(lambda v: "" if v is None or pd.isna(v) else ("yes" if v else "no")),
            "ms": t["latency_ms"].map(lambda v: num(v, "{:.2f}")),
            "source": t["source"].fillna(""),
        })
        st.caption("Outcome uses the known answer from the replay data. "
                   "❌ fraud missed = fraud the model allowed · ⚠️ false alarm = genuine payment flagged. "
                   "Top reasons (SHAP) are worked out for flagged payments. Blank = not available.")
        st.dataframe(table, hide_index=True, use_container_width=True)

live()

# ---- Test a payment --------------------------------------------------------------------------
st.divider()
st.subheader("Test a payment")
st.caption("Fill in what you know; anything left out is treated as unknown, as in training. "
           "Send the same card twice with a bigger amount to see its history change.")
with st.form("test_payment"):
    a, b, c, d = st.columns(4)
    amount = a.number_input("Amount ($)", min_value=0.0, value=117.0, step=10.0)
    product = b.selectbox("Product code", ["W", "H", "C", "S", "R"])
    card1 = c.number_input("Card number group (card1)", min_value=0, value=9500, step=1)
    card2 = d.number_input("Bank code (card2)", min_value=0, value=111, step=1)
    e, f, g, h = st.columns(4)
    card4 = e.selectbox("Card network", ["visa", "mastercard", "american express", "discover"])
    card6 = f.selectbox("Card type", ["debit", "credit"])
    addr1 = g.number_input("Billing area (addr1)", min_value=0, value=204, step=1)
    email = h.text_input("Email domain", "gmail.com")
    device = st.text_input("Device", "iOS Device")
    sent = st.form_submit_button("Score this payment", type="primary")

if sent:
    payload = {
        "transaction_id": f"manual-{datetime.now():%H%M%S}",
        "source": "manual",
        "features": {"TransactionAmt": amount, "ProductCD": product, "card1": card1, "card2": card2,
                     "card4": card4, "card6": card6, "addr1": addr1, "P_emaildomain": email,
                     "DeviceInfo": device or None},
    }
    res, err = post(f"{API_URL}/score?explain=all", payload)
    if err:
        st.error(err)
    else:
        verdict = "🚩 SUSPICIOUS" if res["decision"] == "suspicious" else "✅ ALLOW"
        (st.error if res["decision"] == "suspicious" else st.success)(
            f"**{verdict}** · fraud score {res['fraud_score']:.3f} (threshold {res['threshold']:.3f}) · "
            f"decided in {res['latency_ms']:.2f} ms")
        if res.get("rule"):
            st.warning(f"**Safety rule:** {res['rule']}. Flagged regardless of the model's score, "
                       "and kept out of the card's history.")
        if res.get("reasons"):
            st.markdown("**Why:** the features that pushed this score " +
                        ("up" if res["fraud_score"] >= res["threshold"] else "down") +
                        f" the most (SHAP, {res['explain_ms']:.1f} ms)")
            for r in res["reasons"]:
                arrow = ":orange[▲ raises risk]" if r["impact"] > 0 else ":blue[▼ lowers risk]"
                st.markdown(f"{arrow} · **{r['name']}** = {r['value']} ({r['impact']:+.2f})")
        h = res.get("card_history")
        if h:
            m = st.columns(4)
            m[0].metric("Card's payments in the last hour", h["tx_last_hour"] if h["tx_last_hour"] is not None else "—")
            m[1].metric("Card's usual amount", money(h["avg_amount"]))
            m[2].metric("This amount vs usual", f"{h['amount_vs_avg']:.1f}×" if h["amount_vs_avg"] else "first payment")
            m[3].metric("Device", "new for this card" if h["new_device"] else
                        ("seen before" if h["new_device"] is False else "—"))
        st.caption("Only these 9 of 431 fields were filled in, so the model has far less to go on than "
                   "with a real payment from the replay month.")

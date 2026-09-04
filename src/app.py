
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="OptionRelay Demo",
    page_icon="📈",
    layout="wide",
)

st.title("OptionRelay")
st.subheader("RL-First Options Research & Paper-Trading Workflow")

st.info(
    "Read-only demonstration using sample/redacted data. "
    "No live or paper orders can be placed from this interface."
)

st.header("1. DQN End-of-Day Intent")

intent = {
    "ticker": "MRVL",
    "decision": "ENTER",
    "direction": "LONG_CALL",
    "option_type": "call",
    "target_dte": 21,
    "target_moneyness": 1.02,
    "quantity": 1,
    "modeled_round_trip_cost_pct": 0.05,
}

left, right = st.columns(2)
with left:
    st.metric("Decision", intent["decision"])
    st.metric("Direction", intent["direction"])
    st.metric("Underlying", intent["ticker"])

with right:
    st.metric("Target DTE", f'{intent["target_dte"]} days')
    st.metric("Target Moneyness", intent["target_moneyness"])
    st.metric("Quantity", intent["quantity"])

st.json(intent)

st.header("2. Active-Contract Resolution")

contracts = pd.DataFrame(
    [
        {
            "Contract": "MRVL 2026-09-18 C 150",
            "Type": "Call",
            "DTE": 16,
            "Strike": 150.00,
            "Moneyness": 1.01,
            "Profile distance": 0.03,
            "Rank": 1,
        },
        {
            "Contract": "MRVL 2026-09-25 C 155",
            "Type": "Call",
            "DTE": 23,
            "Strike": 155.00,
            "Moneyness": 1.04,
            "Profile distance": 0.06,
            "Rank": 2,
        },
        {
            "Contract": "MRVL 2026-09-18 C 155",
            "Type": "Call",
            "DTE": 16,
            "Strike": 155.00,
            "Moneyness": 1.04,
            "Profile distance": 0.07,
            "Rank": 3,
        },
    ]
)

st.dataframe(contracts, use_container_width=True, hide_index=True)

st.header("3. Live Quote-Quality Gates")

gates = pd.DataFrame(
    [
        {"Gate": "Valid bid and ask", "Observed": "Bid 4.10 / Ask 4.30", "Result": "PASS"},
        {"Gate": "Quote freshness", "Observed": "8 seconds", "Result": "PASS"},
        {"Gate": "Minimum displayed size", "Observed": "Bid 18 / Ask 22", "Result": "PASS"},
        {"Gate": "Maximum contract debit", "Observed": "$430", "Result": "PASS"},
        {"Gate": "Round-trip spread cost", "Observed": "4.65%", "Result": "PASS"},
    ]
)

st.dataframe(gates, use_container_width=True, hide_index=True)

st.success(
    "Eligible candidate: MRVL 2026-09-18 C 150. "
    "Human authorization is required before any paper-order submission."
)

st.header("Operating Model")

st.write(
    "OptionRelay automates DQN inference, structured intent generation, "
    "active-contract resolution, and quote-quality validation. "
    "Paper-order authorization remains human-supervised."
)

st.caption(
    "This interface displays representative, redacted workflow outputs. "
    "It does not expose credentials or connect to order execution."
)

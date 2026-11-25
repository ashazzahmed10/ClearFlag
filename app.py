"""ClearFlag demo page.

Run:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from clearflag import ClearFlag

st.set_page_config(page_title="ClearFlag", page_icon="✅", layout="centered")
st.title("ClearFlag")
st.caption("Real-time UPI fraud alerts that explain themselves")


@st.cache_resource
def load():
    return ClearFlag("model")


engine = load()

with st.form("payment"):
    c1, c2 = st.columns(2)
    amount = c1.number_input("Amount (₹)", min_value=1.0, value=48000.0, step=1000.0)
    ptype = c2.selectbox("Payment type", ["TRANSFER", "CASH_OUT"])
    hour = c1.slider("Time of payment (hour)", 0, 23, 2)
    sender_bal = c2.number_input("Sender's balance before (₹)", min_value=0.0, value=50000.0, step=1000.0)
    payee_bal = c1.number_input("Payee's balance before (₹)", min_value=0.0, value=0.0, step=1000.0)
    payee_gets = c2.checkbox("Money shows up in payee's balance", value=False)
    go = st.form_submit_button("Check payment")

if go:
    new_sender = max(sender_bal - amount, 0.0)
    new_payee = payee_bal + amount if payee_gets else payee_bal
    pay = pd.DataFrame([{
        "step": 24 + hour, "type": ptype, "amount": amount,
        "oldbalanceOrg": sender_bal, "newbalanceOrig": new_sender,
        "oldbalanceDest": payee_bal, "newbalanceDest": new_payee,
    }])
    r = engine.score(pay)[0]
    colour = {"ALLOW": "green", "VERIFY": "orange", "HOLD": "red"}[r["action"]]
    label = {"ALLOW": "Looks safe: allow", "VERIFY": "Check first: verify with OTP + call",
             "HOLD": "Payment on hold: looks risky"}[r["action"]]
    st.markdown(f"### :{colour}[{label}]")
    st.metric("Risk score", f"{r['risk_score']} / 100")
    if r["reasons"]:
        st.markdown("**Why we flagged it**")
        for i, reason in enumerate(r["reasons"], 1):
            st.markdown(f"{i}. {reason}")

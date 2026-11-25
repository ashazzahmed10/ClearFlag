# ClearFlag

**Real-time UPI fraud alerts that explain themselves.**
Team CodeZephyr · Track 1 (Digital Payments Fraud Detection) · FinTech Innovation Hackathon 2026, Mangaluru

> Early prototype. Built to show the core idea working end to end; more features come at the boot camp.

## The idea

UPI scammers trick people into paying, then move the money out within minutes. Most fraud checks either let a payment through or block it silently. ClearFlag checks each payment **before it goes through**, gives it a **0–100 risk score**, explains **why** in plain words, and picks a graded action:

| Score | Action | What the customer sees |
|---|---|---|
| 0–29 | **ALLOW** | Nothing, payment goes through |
| 30–69 | **VERIFY** | Confirm with OTP + call before paying |
| 70–100 | **HOLD** | Payment paused, reasons shown, bank reviews |

## How it works

1. A payment request comes in (amount, type, time, balances).
2. `clearflag/features.py` turns it into behaviour signals: share of balance sent, payee with no history, late-night timing, balances that don't add up.
3. An XGBoost model scores the payment.
4. TreeSHAP values (from XGBoost's `pred_contribs`) find the signals that pushed the risk up, and `clearflag/engine.py` turns them into plain-language reasons.
5. The score maps to ALLOW / VERIFY / HOLD.

```
Payment ──▶ features ──▶ XGBoost score ──▶ SHAP reasons ──▶ ALLOW / VERIFY / HOLD
```

## Run it

```bash
pip install -r requirements.txt

# 1. Get the data: PaySim from Kaggle ("Synthetic Financial Datasets For Fraud Detection")
#    Put the CSV in data/

# 2. Train (uses earlier hours to train, later hours to test)
python train.py --data data/PS_20174392719_1491204439457_log.csv

# 3a. Demo page
streamlit run app.py

# 3b. Scoring API (docs at http://127.0.0.1:8000/docs)
uvicorn api:app --reload
```

Example API call:

```bash
curl -X POST http://127.0.0.1:8000/score -H "Content-Type: application/json" \
  -d '{"step":26,"type":"TRANSFER","amount":48000,"oldbalanceOrg":50000,"newbalanceOrig":0,"oldbalanceDest":0,"newbalanceDest":0}'
```

```json
{"risk_score": 80, "action": "HOLD",
 "reasons": ["Payee account has no history (looks new or unused)",
             "Money doesn't show up in the payee's balance",
             "Sent late at night (midnight to 6 AM)"]}
```

## Results

`train.py` writes `model/metrics.json` with: fraud caught (%), precision, false alarms per 1,000 genuine payments, and a comparison with PaySim's built-in rule.

<!-- Paste your numbers from model/metrics.json here after training on PaySim. -->

## Data and privacy

- Trained on **PaySim**, a public *synthetic* dataset, so no real customer data is used.
- Uses only payment behaviour (amounts, balances, timing), never PINs, Aadhaar or passwords.

## Roadmap (boot camp, 4–6 Dec)

- Customer alert screen (mock-up in the submission deck) wired to the API
- Bank analyst dashboard for VERIFY and HOLD cases
- Kannada alerts
- Mule-account flags (e.g. RBIH's MuleHunter.AI) as an extra signal

## Project layout

```
clearflag/features.py   behaviour signals
clearflag/engine.py     score, reasons, action
train.py                train + evaluate (time-based split)
api.py                  FastAPI /score endpoint
app.py                  Streamlit demo page
tests/make_fake_data.py tiny FAKE dataset, only to check the code runs
```

## Team

- **Mohd. Ashaz Ahmed** (Leader): ML & back end
- **Musavir Mehdi Mir**: front end, dashboard & UX

DSATM, Bengaluru

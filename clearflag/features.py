"""Turn raw PaySim-style transactions into model features.

PaySim columns: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud.
`step` is one hour of simulated time (1 to 743, about 30 days).
"""
import numpy as np
import pandas as pd

PAYMENT_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

FEATURES = [
    "amount_log",
    "hour",
    "night",
    "share_of_balance",
    "empties_account",
    "payee_was_empty",
    "payee_balance_unchanged",
    "sender_balance_mismatch",
    "is_transfer",
    "is_cash_out",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with exactly the columns in FEATURES."""
    out = pd.DataFrame(index=df.index)
    amount = df["amount"].astype(float)
    old_org = df["oldbalanceOrg"].astype(float)
    new_org = df["newbalanceOrig"].astype(float)
    old_dst = df["oldbalanceDest"].astype(float)
    new_dst = df["newbalanceDest"].astype(float)

    out["amount_log"] = np.log10(amount + 1)
    out["hour"] = (df["step"].astype(int) % 24)
    out["night"] = out["hour"].between(0, 5).astype(int)
    out["share_of_balance"] = np.where(old_org > 0, np.clip(amount / old_org, 0, 10), 10.0)
    out["empties_account"] = ((old_org > 0) & (new_org <= 0.01)).astype(int)
    out["payee_was_empty"] = ((old_dst <= 0.01) & (new_dst <= 0.01)).astype(int)
    out["payee_balance_unchanged"] = (np.abs(new_dst - old_dst) < 0.01).astype(int)
    out["sender_balance_mismatch"] = (np.abs(old_org - amount - new_org) > 1.0).astype(int)
    out["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
    out["is_cash_out"] = (df["type"] == "CASH_OUT").astype(int)
    return out[FEATURES]

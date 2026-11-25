"""Score a payment 0-100, explain it in plain words, and pick an action."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .features import FEATURES, build_features

# Plain-language reason for each feature, used when it pushes the risk UP.
REASONS = {
    "amount_log": lambda r: f"Large amount: ₹{r['amount']:,.0f}",
    "hour": lambda r: f"Unusual time of day ({int(r['step']) % 24:02d}:00)",
    "night": lambda r: "Sent late at night (midnight to 6 AM)",
    "share_of_balance": lambda r: "Sends most of the account balance in one go",
    "empties_account": lambda r: "Leaves the sender's account almost empty",
    "payee_was_empty": lambda r: "Payee account has no history (looks new or unused)",
    "payee_balance_unchanged": lambda r: "Money doesn't show up in the payee's balance",
    "sender_balance_mismatch": lambda r: "Sender's balances don't add up",
    "is_transfer": lambda r: "Transfer to another account (common scam route)",
    "is_cash_out": lambda r: "Cash-out right away (common after a scam)",
}

DEFAULT_THRESHOLDS = {"verify": 30, "hold": 70}


class ClearFlag:
    def __init__(self, model_dir="model"):
        model_dir = Path(model_dir)
        self.booster = xgb.Booster()
        self.booster.load_model(str(model_dir / "model.json"))
        cfg_path = model_dir / "thresholds.json"
        self.thresholds = json.loads(cfg_path.read_text()) if cfg_path.exists() else DEFAULT_THRESHOLDS

    def action(self, score):
        if score >= self.thresholds["hold"]:
            return "HOLD"
        if score >= self.thresholds["verify"]:
            return "VERIFY"
        return "ALLOW"

    def score(self, payments: pd.DataFrame, top_k=3):
        """Return one dict per payment: score, action and top reasons."""
        X = build_features(payments)
        d = xgb.DMatrix(X, feature_names=FEATURES)
        prob = self.booster.predict(d)
        contrib = self.booster.predict(d, pred_contribs=True)[:, :-1]  # TreeSHAP values
        results = []
        for i, (_, row) in enumerate(payments.iterrows()):
            s = int(round(float(prob[i]) * 100))
            order = np.argsort(-contrib[i])
            reasons = [REASONS[FEATURES[j]](row) for j in order[:top_k] if contrib[i, j] > 0]
            results.append({"risk_score": s, "action": self.action(s), "reasons": reasons})
        return results

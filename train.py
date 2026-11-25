"""Train the ClearFlag model on PaySim and report how well it does.

Usage:
    python train.py --data data/PS_20174392719_1491204439457_log.csv

Splits by time (train on earlier hours, test on later hours), so the model
is always tested on payments from the "future".
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_score, recall_score

from clearflag.features import FEATURES, build_features

p = argparse.ArgumentParser()
p.add_argument("--data", required=True, help="Path to the PaySim CSV")
p.add_argument("--split-step", type=int, default=500, help="Hours up to this step are used for training")
p.add_argument("--out", default="model")
args = p.parse_args()

print("Loading", args.data)
df = pd.read_csv(args.data)
# Fraud in PaySim only happens on TRANSFER and CASH_OUT, so we score those.
df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].reset_index(drop=True)

train = df[df["step"] <= args.split_step]
test = df[df["step"] > args.split_step]
print(f"Train: {len(train):,} payments ({train.isFraud.sum():,} fraud) | "
      f"Test: {len(test):,} payments ({test.isFraud.sum():,} fraud)")

Xtr, ytr = build_features(train), train["isFraud"].values
Xte, yte = build_features(test), test["isFraud"].values

ratio = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
model = xgb.XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.8,
    colsample_bytree=0.8, scale_pos_weight=min(ratio, 50), eval_metric="aucpr",
    tree_method="hist", random_state=42,
)
model.fit(Xtr, ytr)

prob = model.predict_proba(Xte)[:, 1]
score = np.round(prob * 100)
report = {"test_payments": int(len(yte)), "test_frauds": int(yte.sum()),
          "average_precision": round(float(average_precision_score(yte, prob)), 4), "by_threshold": {}}
genuine = (yte == 0).sum()
for t in (30, 50, 70):
    flag = score >= t
    report["by_threshold"][t] = {
        "fraud_caught_pct": round(100 * recall_score(yte, flag), 2),
        "precision_pct": round(100 * precision_score(yte, flag, zero_division=0), 2),
        "false_alarms_per_1000_genuine": round(1000 * float((flag & (yte == 0)).sum()) / genuine, 2),
    }

# Baseline: PaySim's built-in rule (flags transfers over 200,000)
base = test["isFlaggedFraud"].values.astype(bool)
report["baseline_rule"] = {
    "fraud_caught_pct": round(100 * recall_score(yte, base), 2),
    "false_alarms_per_1000_genuine": round(1000 * float((base & (yte == 0)).sum()) / genuine, 2),
}

out = Path(args.out)
out.mkdir(exist_ok=True)
model.get_booster().save_model(str(out / "model.json"))
(out / "thresholds.json").write_text(json.dumps({"verify": 30, "hold": 70}))
(out / "metrics.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
print(f"Saved model to {out}/")

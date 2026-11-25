"""Generate a small FAKE PaySim-shaped CSV, only to check the code runs.

Results on this file mean nothing. Use the real PaySim data for real numbers.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 20000
step = rng.integers(1, 744, n)
ptype = rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT"], n, p=[0.3, 0.4, 0.3])
old_org = rng.lognormal(10, 1.2, n)
amount = np.minimum(rng.lognormal(9, 1.3, n), old_org)
fraud = (rng.random(n) < 0.01) & np.isin(ptype, ["TRANSFER", "CASH_OUT"])
amount = np.where(fraud, old_org, amount)  # frauds empty the account
new_org = np.maximum(old_org - amount, 0)
old_dst = np.where(fraud & (rng.random(n) < 0.7), 0, rng.lognormal(10, 1.5, n))
new_dst = np.where(fraud & (old_dst == 0), 0, old_dst + amount)
pd.DataFrame({
    "step": step, "type": ptype, "amount": amount.round(2),
    "nameOrig": [f"C{i}" for i in range(n)], "oldbalanceOrg": old_org.round(2), "newbalanceOrig": new_org.round(2),
    "nameDest": [f"C{i+n}" for i in range(n)], "oldbalanceDest": old_dst.round(2), "newbalanceDest": new_dst.round(2),
    "isFraud": fraud.astype(int), "isFlaggedFraud": ((ptype == "TRANSFER") & (amount > 200000)).astype(int),
}).to_csv("tests/fake_paysim.csv", index=False)
print("wrote tests/fake_paysim.csv")

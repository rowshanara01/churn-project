"""Real data nai? Realistic synthetic churn data banai (missing value + imbalance soho)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 3000

tenure = rng.integers(1, 73, n)
contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
internet = rng.choice(["DSL", "Fiber", "None"], n, p=[0.35, 0.45, 0.20])
payment = rng.choice(["Card", "Bank transfer", "Mobile banking", "Cash"], n)
age = rng.integers(18, 80, n)
support_calls = rng.poisson(1.5, n)
base = {"DSL": 40, "Fiber": 75, "None": 20}
monthly = np.array([base[i] for i in internet]) + rng.normal(0, 8, n)
monthly = monthly.clip(10, 130).round(2)
total = (monthly * tenure * rng.normal(1, 0.03, n)).round(2)

# churn er "hidden rule" (model eta khuje ber korbe)
z = (
    -3.3
    - 0.05 * tenure
    + 0.02 * monthly
    + 0.45 * support_calls
    + 1.3 * (contract == "Month-to-month")
    - 0.9 * (contract == "Two year")
    + 0.6 * (internet == "Fiber")
    + 0.5 * (payment == "Cash")
    + rng.normal(0, 0.8, n)
)
churn = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

df = pd.DataFrame({
    "tenure": tenure, "monthly_charges": monthly, "total_charges": total,
    "age": age, "support_calls": support_calls, "contract": contract,
    "internet_service": internet, "payment_method": payment, "churn": churn,
})
# kichu missing value (real data te thake)
df.loc[rng.choice(n, 120, replace=False), "total_charges"] = np.nan
df.loc[rng.choice(n, 80, replace=False), "age"] = np.nan

df.to_csv("data/churn.csv", index=False)
print("Rows:", len(df), "| Churn rate:", round(df.churn.mean(), 3))

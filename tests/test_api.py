from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
RISKY = {"tenure": 2, "monthly_charges": 95.0, "support_calls": 6, "contract": "Month-to-month",
         "internet_service": "Fiber", "payment_method": "Cash"}
SAFE = {"tenure": 60, "monthly_charges": 30.0, "support_calls": 0, "contract": "Two year",
        "internet_service": "DSL", "payment_method": "Card", "age": 45, "total_charges": 1800}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_risky_customer_scores_higher_than_safe():
    risky = client.post("/predict", json=RISKY).json()
    safe = client.post("/predict", json=SAFE).json()
    assert risky["churn_probability"] > safe["churn_probability"]
    assert risky["will_churn"] is True and safe["will_churn"] is False


def test_missing_optional_fields_ok():
    assert client.post("/predict", json=RISKY).status_code == 200


def test_validation_rejects_bad_input():
    bad = {**RISKY, "contract": "Weekly"}
    assert client.post("/predict", json=bad).status_code == 422


def test_batch():
    r = client.post("/predict-batch", json=[RISKY, SAFE]).json()
    assert r["count"] == 2

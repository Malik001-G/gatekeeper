"""API tests with a tiny stand-in model, so they run without the real model files."""
import json

import lightgbm as lgb
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    rng = np.random.default_rng(0)
    n = 2000
    amt = rng.exponential(100, n)
    prod = rng.integers(0, 3, n)
    y = ((amt > 250) & (prod == 2)).astype(int)
    X = np.column_stack([amt, prod]).astype("float32")
    booster = lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(X, y), num_boost_round=20)
    booster.save_model(str(d / "lightgbm_model.txt"))
    (d / "preprocess.json").write_text(json.dumps({
        "features": ["TransactionAmt", "ProductCD"], "text_cols": ["ProductCD"],
        "codes": {"ProductCD": {"W": 0, "H": 1, "C": 2}},
        "missing_value": -999, "unseen_text": -2, "missing_text": -1}))
    (d / "champion.json").write_text(json.dumps({"model": "LightGBM", "threshold": 0.5,
                                                 "threshold_policy": "test"}))

    import app.main as main
    main.MODEL_DIR = str(d)
    # Run without Redis and Postgres, even inside the api container where both are configured,
    # so tests never write into the real card history or decision log.
    main.REDIS_URL = None
    main.DATABASE_URL = None
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    r = client.get("/health").json()
    assert r["model_loaded"] is True


def test_model_info(client):
    r = client.get("/model").json()
    assert r["features"] == 2 and r["threshold"] == 0.5


def test_suspicious_payment(client):
    r = client.post("/score", json={"transaction_id": "t1",
                                    "features": {"TransactionAmt": 900, "ProductCD": "C"}}).json()
    assert r["decision"] == "suspicious" and r["fraud_score"] > 0.5 and r["transaction_id"] == "t1"


def test_normal_payment(client):
    r = client.post("/score", json={"features": {"TransactionAmt": 20, "ProductCD": "W"}}).json()
    assert r["decision"] == "allow"


def test_missing_and_unknown_fields(client):
    r = client.post("/score", json={"features": {"ProductCD": "never-seen", "extra_field": 1}})
    assert r.status_code == 200
    assert r.json()["features_received"] == 2


def test_bad_number_is_treated_as_missing(client):
    r = client.post("/score", json={"features": {"TransactionAmt": "abc"}})
    assert r.status_code == 200


def test_vector_matches_prepare(client):
    """Single-payment fast path must give exactly the same inputs as the batch path."""
    import pandas as pd
    import app.main as main
    champ = main.state["champion"]
    payments = [{"TransactionAmt": 120.5, "ProductCD": "H"}, {"ProductCD": None},
                {"TransactionAmt": "abc", "ProductCD": "zzz"}, {}]
    for p in payments:
        clean = {k: (v if k == "ProductCD" else pd.to_numeric(v, errors="coerce")) for k, v in p.items()}
        batch = champ.prepare(pd.DataFrame([clean]).reindex(columns=champ.features))
        assert (champ.vector(p) == batch).all(), p


def test_score_without_history_or_store(client):
    r = client.post("/score", json={"features": {"TransactionAmt": 20, "card1": 1}}).json()
    assert r["card_history"] is None and r["model_ms"] <= r["latency_ms"]
    assert client.get("/stats").status_code == 503


def test_huge_amount_is_flagged_by_rule(client):
    r = client.post("/score", json={"features": {"TransactionAmt": 118888007988899, "ProductCD": "W"}}).json()
    assert r["decision"] == "suspicious" and r["rule"] and "outside the range" in r["rule"]


def test_negative_amount_is_flagged_by_rule(client):
    r = client.post("/score", json={"features": {"TransactionAmt": -5}}).json()
    assert r["decision"] == "suspicious" and r["rule"] == "amount must be positive"


def test_normal_amount_has_no_rule(client):
    r = client.post("/score", json={"features": {"TransactionAmt": 20, "ProductCD": "W"}}).json()
    assert r["rule"] is None and r["decision"] == "allow"


def test_flagged_payment_has_reasons(client):
    r = client.post("/score", json={"features": {"TransactionAmt": 900, "ProductCD": "C"}}).json()
    assert r["decision"] == "suspicious"
    assert 1 <= len(r["reasons"]) <= 3 and r["explain_ms"] is not None
    assert all(x["effect"] == "raises risk" and x["impact"] > 0 for x in r["reasons"])
    names = {x["name"] for x in r["reasons"]}
    assert names <= {"Payment amount", "Product type"}
    amount = next(x for x in r["reasons"] if x["feature"] == "TransactionAmt")
    assert amount["value"] == "$900.00" and amount["text"] == "Payment amount ($900.00) raises risk"


def test_allowed_payment_has_no_reasons_by_default(client):
    r = client.post("/score", json={"features": {"TransactionAmt": 20, "ProductCD": "W"}}).json()
    assert r["decision"] == "allow" and r["reasons"] is None and r["explain_ms"] is None


def test_explain_all_gives_reasons_for_allowed(client):
    r = client.post("/score?explain=all", json={"features": {"TransactionAmt": 20, "ProductCD": "W"}}).json()
    assert r["decision"] == "allow" and r["reasons"]
    assert all(x["effect"] == "lowers risk" and x["impact"] < 0 for x in r["reasons"])


def test_explain_none(client):
    r = client.post("/score?explain=none", json={"features": {"TransactionAmt": 900, "ProductCD": "C"}}).json()
    assert r["decision"] == "suspicious" and r["reasons"] is None


def test_shap_values_add_up_to_the_score(client):
    """SHAP values plus the base value must give back the model's raw score exactly."""
    import app.main as main
    champ = main.state["champion"]
    x = champ.vector({"TransactionAmt": 400, "ProductCD": "C"})
    contrib = champ.booster.predict(x, pred_contrib=True)[0]
    raw = champ.booster.predict(x, raw_score=True)[0]
    assert abs(contrib.sum() - raw) < 1e-6

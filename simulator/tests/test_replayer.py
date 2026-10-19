import time

import numpy as np
import pandas as pd
import pytest

from sim.replayer import Replayer, to_payload


@pytest.fixture
def df():
    n = 50
    return pd.DataFrame({
        "TransactionID": np.arange(n) + 1, "TransactionDT": np.arange(n)[::-1] * 60 + 86400,
        "isFraud": [1 if i % 10 == 0 else 0 for i in range(n)],
        "TransactionAmt": np.linspace(10, 500, n), "card1": 1000.0, "V1": np.nan, "ProductCD": "W",
    }).sort_values("TransactionDT").reset_index(drop=True)


def fake_api(threshold=250):
    sent = []
    def send(p):
        sent.append(p)
        return {"decision": "suspicious" if p["features"]["TransactionAmt"] > threshold else "allow"}
    return send, sent


def test_payload_drops_missing_and_ids(df):
    p = to_payload(df.iloc[0].to_dict())
    assert "V1" not in p["features"] and "isFraud" not in p["features"] and "TransactionDT" not in p["features"]
    assert p["event_time"].startswith("2017-12-02") and p["source"] == "simulator" and p["label"] in (0, 1)


def test_sends_in_time_order_and_counts(df):
    send, sent = fake_api()
    rep = Replayer(df, send)
    rep.start(rate=0)
    rep.wait()
    times = [p["event_time"] for p in sent]
    assert times == sorted(times) and len(sent) == 50
    s = rep.progress.as_dict()
    assert s["sent"] == 50 and s["frauds"] == 5 and not s["running"]
    assert s["flagged"] == sum(p["features"]["TransactionAmt"] > 250 for p in sent)


def test_limit_start_and_rate(df):
    send, sent = fake_api()
    rep = Replayer(df, send)
    t0 = time.time()
    rep.start(rate=50, limit=10, start=5)
    rep.wait()
    assert len(sent) == 10 and time.time() - t0 >= 0.15   # 10 payments at 50/s take ~0.2 s


def test_stop(df):
    send, sent = fake_api()
    rep = Replayer(df, send)
    rep.start(rate=20)
    time.sleep(0.3)
    rep.stop()
    assert 0 < len(sent) < 50 and not rep.running


def test_errors_are_counted(df):
    def send(p):
        raise ConnectionError("api down")
    rep = Replayer(df, send)
    rep.start(rate=0, limit=5)
    rep.wait()
    s = rep.progress.as_dict()
    assert s["errors"] == 5 and s["sent"] == 0 and "api down" in s["last_error"]


def test_parallel_workers(df):
    send, sent = fake_api()
    rep = Replayer(df, send)
    rep.start(rate=0, workers=4)
    rep.wait()
    assert rep.progress.as_dict()["sent"] == 50

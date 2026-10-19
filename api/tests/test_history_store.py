"""Card history (Redis) and decision log (Postgres)."""
import os
import time

import fakeredis
import pytest

from gatekeeper.api.app.history import CardHistory, card_key


@pytest.fixture
def history():
    h = CardHistory("redis://localhost:6379/0")
    h.r = fakeredis.FakeRedis()
    return h


def test_card_key():
    assert card_key({"card1": 9500.0, "card2": 111.0, "addr1": 204.0}) == "9500|111|204"
    assert card_key({"card1": 9500, "addr1": None}) == "9500||"
    assert card_key({"TransactionAmt": 10}) is None


def test_first_payment_has_no_average(history):
    s = history.observe({"card1": 1, "TransactionAmt": 50, "DeviceInfo": "iOS"}, event_ts=1000)
    assert s.tx_last_hour == 1 and s.avg_amount is None and s.amount_vs_avg is None and s.new_device is True


def test_history_builds_up(history):
    f = {"card1": 1, "card2": 2, "addr1": 3}
    history.observe({**f, "TransactionAmt": 50, "DeviceInfo": "iOS"}, event_ts=1000)
    history.observe({**f, "TransactionAmt": 150, "DeviceInfo": "iOS"}, event_ts=1600)
    s = history.observe({**f, "TransactionAmt": 600, "DeviceInfo": "Windows"}, event_ts=2000)
    assert s.tx_last_hour == 3
    assert s.avg_amount == 100.0          # average of the two earlier payments
    assert s.amount_vs_avg == 6.0         # this one is 6x the card's usual amount
    assert s.new_device is True           # Windows not seen on this card before


def test_old_payments_leave_the_hour_window(history):
    f = {"card1": 7, "TransactionAmt": 10}
    history.observe(f, event_ts=0)
    history.observe(f, event_ts=100)
    s = history.observe(f, event_ts=100 + 3600 + 1)
    assert s.tx_last_hour == 1


def test_seen_device_is_not_new(history):
    f = {"card1": 5, "TransactionAmt": 10, "DeviceInfo": "iOS"}
    history.observe(f, event_ts=1)
    assert history.observe(f, event_ts=2).new_device is False


@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="needs TEST_DATABASE_URL")
def test_decision_store_roundtrip():
    from gatekeeper.api.app.store import DecisionStore
    store = DecisionStore(os.environ["TEST_DATABASE_URL"])
    for _ in range(50):
        if store.ready:
            break
        time.sleep(0.1)
    with store.pool.connection() as conn:
        conn.execute("TRUNCATE decisions")
    for i, (label, decision, amt) in enumerate([(1, "suspicious", 100), (1, "allow", 40),
                                                (0, "allow", 10), (0, "suspicious", 20), (None, "allow", 5)]):
        store.add({"transaction_id": f"t{i}", "fraud_score": 0.5, "decision": decision, "threshold": 0.16,
                   "amount": amt, "latency_ms": 1.0, "label": label})
    store.flush()
    s = store.stats()
    assert s["decisions"] == 5 and s["flagged"] == 2 and s["frauds"] == 2 and s["frauds_caught"] == 1
    assert s["false_alarms"] == 1 and s["fraud_missed_usd"] == 40 and s["fraud_stopped_usd"] == 100
    assert [r["transaction_id"] for r in store.recent(2)] == ["t4", "t3"]
    store.close()

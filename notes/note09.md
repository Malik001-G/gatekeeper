# Dashboard and a safety rule

Code: `dashboard/app.py` (Streamlit + Altair, port 8501), `api/app/main.py` (safety rule).

## Dashboard
- **Sidebar:** start, stop and speed controls for the replay; a button to clear card history;
  a results window (all time or the last N minutes).
- **Live results**, refreshed every 2 seconds: payments scored, flagged, fraud caught,
  false alarms, fraud stopped vs missed in $, and decision time.
- **Fraud score chart:** the latest payments' scores against the threshold line
  (allow in blue, suspicious in orange).
- **Latest flagged payments**, with their card history.
- **All latest decisions** table with plain column names and an outcome per payment:
  ✅ fraud caught, ❌ fraud missed, ⚠️ false alarm, ✓ genuine allowed.
  Empty cells mean the value is not available (for example, no device info).
- **Test a payment:** enter amount, product, card and email domain and get a decision instantly.

## Bug found while testing: an impossible amount was allowed
A test payment of $118,888,007,988,899 got a fraud score of 0.102 and was **allowed**.
It also entered the card's history, so that card's "usual amount" jumped to $39 million.

Why: tree models cannot extrapolate. An amount far beyond anything in training lands in the same
leaf as the largest amounts the model did see, so it gets an ordinary score.

Fix: a safety rule in front of the model.
- Amounts that are zero, negative or above `MAX_AMOUNT` (default $50,000) are marked
  **suspicious** whatever the score.
- The response's `rule` field says why; the rule is saved in the decision log and shown on
  the dashboard.
- These payments are kept out of the card's history.

3 new API tests (huge amount, negative amount, normal amount).

## What the decisions table shows
Many missed frauds are small, normal-looking payments. Example from the replay:
$8.40, product C, fraud score 0.009, card's usual amount $24.70 (0.34×), 3 payments in the last hour.
Nothing about it stands out to the model or the card history.

Known limitation: by count the champion catches about half of fraud, but by value it stops less
(about 42% of fraud dollars), because the frauds it misses are larger on average.

## Next
SHAP: the top 3 reasons behind each decision, in plain language.

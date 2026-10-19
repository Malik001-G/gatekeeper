"""Plain-language reasons behind a decision, from SHAP values.

LightGBM computes exact TreeSHAP values itself (`pred_contrib=True`): one number per feature
saying how far that feature pushed this payment's score up or down, compared with an average
payment. The values plus the average add up exactly to the model's raw score (log-odds).

Many IEEE-CIS columns are anonymised by the data provider (C, D, M, V, id_ groups), so their
plain names describe the group, not an exact meaning.
"""
from __future__ import annotations

import re
from typing import Optional

NAMES = {
    "TransactionAmt": "Payment amount",
    "ProductCD": "Product type",
    "card1": "Card number group",
    "card2": "Card issuer code",
    "card3": "Card issuer country code",
    "card4": "Card network",
    "card5": "Card category code",
    "card6": "Card type",
    "addr1": "Billing area",
    "addr2": "Billing country",
    "dist1": "Distance between addresses",
    "dist2": "Second address distance",
    "P_emaildomain": "Payer's email domain",
    "R_emaildomain": "Recipient's email domain",
    "DeviceType": "Device type",
    "DeviceInfo": "Device",
}

GROUPS = [
    (r"C(\d+)", "Linked count C{}"),       # e.g. how many addresses are linked to the card
    (r"D(\d+)", "Days since D{}"),         # e.g. days since the previous payment
    (r"M(\d+)", "Match check M{}"),        # e.g. name on card matches address
    (r"V(\d+)", "Risk signal V{}"),        # features engineered by the data provider
    (r"id_(\d+)", "Identity id_{}"),       # device and connection details
]


def plain_name(feature: str) -> str:
    if feature in NAMES:
        return NAMES[feature]
    for pattern, name in GROUPS:
        m = re.fullmatch(pattern, feature)
        if m:
            return name.format(m.group(1))
    return feature


def show_value(feature: str, value) -> str:
    if value is None or value != value:
        return "not given"
    if feature == "TransactionAmt":
        try:
            return f"${float(value):,.2f}"
        except (TypeError, ValueError):
            return "not given"
    if isinstance(value, (int, float)):
        v = float(value)
        return str(int(v)) if v.is_integer() else f"{v:.4g}"
    return str(value)


def top_reasons(contrib, features: list[str], raw: dict, suspicious: bool, k: int = 3) -> list[dict]:
    """The k features that pushed hardest towards the decision that was made.

    contrib: SHAP values for one payment, one per feature (the base value already removed).
    For a suspicious payment these are what raised the risk; for an allowed one, what lowered it.
    """
    sign = 1.0 if suspicious else -1.0
    order = sorted(range(len(features)), key=lambda j: sign * contrib[j], reverse=True)
    out = []
    for j in order[:k]:
        impact = float(contrib[j])
        if sign * impact <= 0:  # nothing else pushes this way
            break
        f = features[j]
        name, value = plain_name(f), show_value(f, raw.get(f))
        effect = "raises risk" if impact > 0 else "lowers risk"
        out.append({"feature": f, "name": name, "value": value, "effect": effect,
                    "impact": round(impact, 4), "text": f"{name} ({value}) {effect}"})
    return out


def reasons_text(reasons: Optional[list[dict]]) -> str:
    return "; ".join(r["text"] for r in reasons or [])

from app.reasons import plain_name, show_value, top_reasons


def test_plain_names():
    assert plain_name("TransactionAmt") == "Payment amount"
    assert plain_name("P_emaildomain") == "Payer's email domain"
    assert plain_name("C13") == "Linked count C13"
    assert plain_name("V258") == "Risk signal V258"
    assert plain_name("id_31") == "Identity id_31"
    assert plain_name("something_else") == "something_else"


def test_values():
    assert show_value("TransactionAmt", 8.5) == "$8.50"
    assert show_value("card1", 9500.0) == "9500"
    assert show_value("C1", None) == "not given"
    assert show_value("P_emaildomain", "gmail.com") == "gmail.com"


def test_top_reasons_follow_the_decision():
    feats = ["a", "b", "c", "d"]
    contrib = [0.5, -0.9, 0.2, -0.1]
    up = top_reasons(contrib, feats, {}, suspicious=True)
    assert [r["feature"] for r in up] == ["a", "c"]          # only features that raised risk
    down = top_reasons(contrib, feats, {}, suspicious=False)
    assert [r["feature"] for r in down] == ["b", "d"]
    assert down[0]["text"] == "b (not given) lowers risk"
def test_long_floats_are_shortened():
    assert show_value("V258", 0.9952423572540283) == "0.9952"
    assert show_value("D1", 14.0) == "14"

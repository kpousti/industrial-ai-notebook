"""Tests: the data is realistic enough to be hard, and the model earns its place."""

from triage import QUEUES, evaluate, keyword_route, make_tickets


def test_tickets_cover_all_queues_with_some_noisy_labels():
    df = make_tickets()
    assert set(df["queue"]) == set(QUEUES)
    mislabeled = (df["queue"] != df["true_queue"]).mean()
    assert 0.02 < mislabeled < 0.10  # some mis-filed history, but not most of it


def test_keyword_rules_route_obvious_tickets():
    assert keyword_route("vpn keeps dropping") == "network"
    assert keyword_route("plc not responding to the hmi") == "shop_floor"


def test_model_beats_keyword_rules_and_stays_under_ceiling():
    results, _, ceiling, _, _ = evaluate()
    ml, kw = results.loc["tfidf + logreg", "accuracy"], results.loc["keyword rules", "accuracy"]
    assert ml > kw + 0.1
    assert ml <= ceiling  # can't beat the label noise; if it does, something leaked


def test_higher_threshold_automates_less_but_more_accurately():
    _, thresholds, _, _, _ = evaluate()
    assert thresholds.loc[0.9, "auto_routed_share"] < thresholds.loc[0.5, "auto_routed_share"]
    assert thresholds.loc[0.9, "auto_routed_accuracy"] >= thresholds.loc[0.5, "auto_routed_accuracy"]

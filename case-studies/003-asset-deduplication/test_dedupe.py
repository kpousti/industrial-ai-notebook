"""Tests: blocking is safe, grouping is transitive, and fuzzy matching earns its place."""

from itertools import combinations

from dedupe import candidate_pairs, cluster, evaluate, make_inventory


def test_blocking_keeps_every_true_duplicate_pair():
    df = make_inventory()
    pairs = candidate_pairs(df)
    expected = sum(
        len(list(combinations(g.index, 2))) for _, g in df.groupby("device_id")
    )
    assert pairs["is_match"].sum() == expected


def test_cluster_is_transitive():
    groups = cluster([1, 2, 3, 4], [(1, 2), (2, 3)])
    assert groups[1] == groups[2] == groups[3]
    assert groups[4] != groups[1]


def test_exact_rule_is_precise_but_misses_most_duplicates():
    results, *_ = evaluate()
    rule = results.loc["exact-key rule"]
    assert rule["precision"] == 1.0
    assert rule["recall"] < 0.6


def test_similarity_model_rebuilds_more_devices_exactly():
    results, *_ = evaluate()
    rule, ml = results.loc["exact-key rule"], results.loc["similarity + logreg"]
    assert ml["f1"] > rule["f1"]
    assert ml["exactly_right"] > rule["exactly_right"]
    assert ml["devices_found"] >= ml["true_devices"]  # under-merging, not over-merging

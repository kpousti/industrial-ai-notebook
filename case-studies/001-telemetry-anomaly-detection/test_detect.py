"""Sanity tests: the data has faults, and each detector finds some without flooding."""

from detect import isolation_forest_flags, make_telemetry, rolling_zscore_flags, score


def test_synthetic_data_has_labeled_faults():
    df = make_telemetry()
    assert 0 < df["is_fault"].sum() < len(df) * 0.1  # faults exist but are rare


def test_zscore_is_precise_on_spikes():
    df = make_telemetry()
    assert score(df, rolling_zscore_flags(df))["precision"] >= 0.8


def test_isolation_forest_catches_more_faults_than_baseline():
    df = make_telemetry()
    z = score(df, rolling_zscore_flags(df))
    iso = score(df, isolation_forest_flags(df))
    assert iso["recall"] > z["recall"]

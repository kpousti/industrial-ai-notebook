"""Case study 003: finding duplicate devices across asset inventories.

The same machine shows up in several systems (an asset register, a network
scan, a purchasing list), each spelling it a little differently. We compare:
  1. Exact-key rule       - merge records only if serial number or MAC match exactly
  2. Similarity + logreg  - score each pair on fuzzy similarities, learn the weights,
                            then group matched pairs into devices (union-find)

All data is synthetic. Run:  python dedupe.py
"""

import re
from difflib import SequenceMatcher
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

SITES = ["north", "south", "east", "west", "central", "harbor"]
MODELS = ["WS-7040", "WS-7060", "LT-5420", "HMI-15T", "PLC-X200", "SCN-300"]
SOURCES = ["register", "netscan", "purchasing"]


# ---------- synthetic data ----------

def _typo(s: str, rng) -> str:
    """One realistic slip: swap two neighbours, or confuse O/0 or I/1."""
    if len(s) < 4:
        return s
    i = int(rng.integers(1, len(s) - 1))
    if rng.random() < 0.5:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    return s.translate(str.maketrans("O0I1", "0O1I"))


def _record(dev: dict, source: str, rng) -> dict:
    """How one source writes down a device. Each source has its own habits."""
    host, serial, mac = dev["hostname"], dev["serial"], dev["mac"]
    if source == "register":   # typed by hand: typos, sometimes lower case
        host = _typo(host, rng) if rng.random() < 0.3 else host
        host = host.lower() if rng.random() < 0.3 else host
        serial = _typo(serial, rng) if rng.random() < 0.15 else serial
        mac = None if rng.random() < 0.5 else mac
    elif source == "netscan":  # sees the network, not the label: no serial, dashed MAC
        host = host.lower() + ".plant.local"
        serial = None
        mac = mac.replace(":", "-").upper()
    else:                      # purchasing: serial from the invoice, no host or MAC
        host = None if rng.random() < 0.6 else host
        mac = None
    return {"device_id": dev["device_id"], "source": source, "site": dev["site"],
            "model": dev["model"], "hostname": host, "serial": serial, "mac": mac}


def make_inventory(n_devices: int = 360, seed: int = 11) -> pd.DataFrame:
    """Each true device appears in 1-3 sources. Hostnames are sequential per site,
    so different devices often look alike (WS-0012 vs WS-0021): the hard part."""
    rng = np.random.default_rng(seed)
    devices, counters = [], {}
    for i in range(n_devices):
        site = SITES[i % len(SITES)]
        model = rng.choice(MODELS)
        prefix = f"{site[:3].upper()}-{model.split('-')[0]}"
        counters[prefix] = counters.get(prefix, 0) + int(rng.integers(1, 4))  # unique, with gaps
        devices.append({
            "device_id": i, "site": site, "model": model,
            "hostname": f"{prefix}-{counters[prefix]:04d}",
            "serial": "SN" + "".join(rng.choice(list("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"), 8)),
            "mac": ":".join(f"{b:02x}" for b in rng.integers(0, 256, 6)),
        })
    rows = []
    for dev in devices:
        k = rng.choice([1, 2, 3], p=[0.25, 0.4, 0.35])
        for src in rng.choice(SOURCES, size=k, replace=False):
            rows.append(_record(dev, src, rng))
    return pd.DataFrame(rows).reset_index(names="record_id")


# ---------- matching ----------

def _norm_host(h):
    return None if h is None else h.split(".")[0].upper()


def _norm_mac(m):
    return None if m is None else re.sub(r"[^0-9a-f]", "", m.lower())


def _sim(a, b) -> float:
    """String similarity 0..1; -1 means 'one side is missing' (kept as its own signal)."""
    if a is None or b is None:
        return -1.0
    return SequenceMatcher(None, a, b).ratio()


def candidate_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Blocking: only compare records at the same site with the same model.
    A device never changes site or model, so this loses no true duplicate."""
    df = df.astype(object).where(df.notna(), None)  # pandas stores None as NaN; undo that
    rows = []
    for _, block in df.groupby(["site", "model"]):
        recs = block.to_dict("records")
        for a, b in combinations(recs, 2):
            if a["source"] == b["source"]:
                continue  # a source doesn't list the same device twice
            rows.append({
                "a": a["record_id"], "b": b["record_id"], "site": a["site"],
                "host_sim": _sim(_norm_host(a["hostname"]), _norm_host(b["hostname"])),
                "host_eq": -1.0 if a["hostname"] is None or b["hostname"] is None
                           else float(_norm_host(a["hostname"]) == _norm_host(b["hostname"])),
                "serial_sim": _sim(a["serial"], b["serial"]),
                "mac_eq": -1.0 if a["mac"] is None or b["mac"] is None
                          else float(_norm_mac(a["mac"]) == _norm_mac(b["mac"])),
                "is_match": int(a["device_id"] == b["device_id"]),
            })
    return pd.DataFrame(rows)


def exact_key_rule(pairs: pd.DataFrame) -> pd.Series:
    """Baseline: same serial (exact) or same MAC once formatting is normalized."""
    return (pairs["serial_sim"] == 1.0) | (pairs["mac_eq"] == 1.0)


FEATURES = ["host_sim", "host_eq", "serial_sim", "mac_eq"]


def _design(pairs: pd.DataFrame) -> pd.DataFrame:
    """Split each feature into 'value when present' + 'is missing' so the model
    can treat 'no serial to compare' differently from 'serials disagree'."""
    X = pd.DataFrame(index=pairs.index)
    for f in FEATURES:
        X[f] = pairs[f].clip(lower=0)
        X[f + "_missing"] = (pairs[f] < 0).astype(float)
    return X


def train_matcher(train_pairs: pd.DataFrame) -> LogisticRegression:
    # Only 8 features and thousands of pairs, so regularize lightly (C=100).
    # With the default C=1 the weights shrink so much that even an exact serial
    # match no longer clears the bar, and the model misses pairs the rule catches.
    model = LogisticRegression(max_iter=1000, C=100)
    model.fit(_design(train_pairs), train_pairs["is_match"])
    return model


def cluster(record_ids, matched_pairs) -> dict:
    """Union-find: if A~B and B~C, all three are one device."""
    parent = {r: r for r in record_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in matched_pairs:
        parent[find(a)] = find(b)
    return {r: find(r) for r in record_ids}


def score(pairs, pred, df, test_records) -> dict:
    groups = cluster(test_records, pairs.loc[pred.values, ["a", "b"]].itertuples(index=False))
    sub = df[df["record_id"].isin(test_records)].assign(group=lambda d: d["record_id"].map(groups))
    true_devices = sub["device_id"].nunique()
    # A device is "exactly right" if its group holds all of its records and nothing else
    exact = sum(sub.groupby("group")["device_id"].nunique().eq(1)
                & sub.groupby("group")["device_id"].first().map(sub["device_id"].value_counts())
                  .eq(sub.groupby("group").size()))
    return {
        "precision": round(precision_score(pairs["is_match"], pred), 3),
        "recall": round(recall_score(pairs["is_match"], pred), 3),
        "f1": round(f1_score(pairs["is_match"], pred), 3),
        "devices_found": len(set(groups.values())),
        "true_devices": true_devices,
        "exactly_right": round(exact / true_devices, 2),
    }


def evaluate(seed: int = 11):
    df = make_inventory(seed=seed)
    pairs = candidate_pairs(df)
    # Train on three sites, test on the other three: the model never sees test devices
    train_sites = SITES[:3]
    train, test = pairs[pairs["site"].isin(train_sites)], pairs[~pairs["site"].isin(train_sites)]
    test_records = df.loc[~df["site"].isin(train_sites), "record_id"].tolist()

    model = train_matcher(train)
    ml_pred = pd.Series(model.predict(_design(test)) == 1, index=test.index)
    results = pd.DataFrame({
        "exact-key rule": score(test, exact_key_rule(test), df, test_records),
        "similarity + logreg": score(test, ml_pred, df, test_records),
    }).T
    results[["devices_found", "true_devices"]] = results[["devices_found", "true_devices"]].astype(int)
    n_all = len(df) * (len(df) - 1) // 2
    return results, len(df), n_all, len(pairs)


if __name__ == "__main__":
    results, n_records, n_all, n_pairs = evaluate()
    print(f"records={n_records}  all pairs={n_all}  after blocking={n_pairs} "
          f"({n_all / n_pairs:.0f}x fewer)\n")
    print(results.to_string())

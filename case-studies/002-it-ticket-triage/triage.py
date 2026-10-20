"""Case study 002: routing IT tickets to the right support queue.

Compares two ways to pick a queue from a ticket's free-text description:
  1. Keyword rules            - the hand-written "if it says VPN, send to Network" approach
  2. TF-IDF + logistic regr.  - learns which words (and word pairs) point to which queue

Then uses the model's confidence to decide which tickets can be auto-routed
and which should go to a human dispatcher.

All data is synthetic. Run:  python triage.py
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

QUEUES = ["network", "access", "endpoint", "application", "shop_floor"]

# Phrases a user might write for each queue. Several deliberately share words
# with other queues ("login", "screen", "connect"), which is what makes this hard.
PHRASES = {
    "network": [
        "vpn keeps dropping", "no internet in building 3", "wifi is very slow",
        "cannot connect to the shared drive", "switch port is down",
        "network cable unplugged warning", "lost connection to the server",
        "packet loss on the warehouse wifi",
    ],
    "access": [
        "account is locked", "need access to the quality folder",
        "password expired and reset link not working", "cannot login after password change",
        "new hire needs an account", "please remove access for contractor",
        "mfa code not arriving", "permission denied on the reports site",
    ],
    "endpoint": [
        "laptop will not turn on", "screen is flickering", "keyboard not working",
        "docking station not detecting monitor", "laptop battery drains fast",
        "need a new mouse", "pc is very slow after update", "blue screen on startup",
    ],
    "application": [
        "erp report shows wrong totals", "email client crashes on open",
        "spreadsheet macro error", "work order app freezes when saving",
        "cannot login to the timesheet app", "error 500 on the purchasing page",
        "calendar invites not syncing", "label printing app shows blank preview",
    ],
    "shop_floor": [
        "hmi screen frozen on line 2", "plc not responding to the hmi",
        "barcode scanner on the packing line not reading",
        "historian not collecting tags from the press", "operator panel cannot login",
        "line 4 terminal lost connection to the controller",
        "badge reader at the cell is offline", "machine data not showing on the dashboard",
    ],
}
FILLER = ["", "please help", "urgent", "since this morning", "again",
          "this is blocking my work", "tried restarting already", "thanks"]


def _messy(phrase: str, rng) -> str:
    """Make a phrase look typed in a hurry: drop some words, swap letters in a few."""
    words = [w for w in phrase.split() if rng.random() > 0.25] or phrase.split()[:1]
    out = []
    for w in words:
        if len(w) > 3 and rng.random() < 0.08:
            i = rng.integers(0, len(w) - 1)
            w = w[:i] + w[i + 1] + w[i] + w[i + 2:]  # swap two adjacent letters
        out.append(w)
    return " ".join(out)


def make_tickets(n: int = 1500, seed: int = 42, distractor_rate: float = 0.35,
                 label_noise: float = 0.06) -> pd.DataFrame:
    """Generate tickets: 1-2 messy phrases from the true queue, sometimes a misleading
    phrase from another queue (real tickets often mention side symptoms), plus filler.
    A small share of labels is then flipped, because historical tickets are sometimes
    closed in the wrong queue; that caps the best achievable accuracy."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        queue = rng.choice(QUEUES, p=[0.25, 0.25, 0.2, 0.18, 0.12])
        parts = [_messy(p, rng) for p in rng.choice(PHRASES[queue], size=rng.integers(1, 3), replace=False)]
        if rng.random() < distractor_rate:
            other = rng.choice([q for q in QUEUES if q != queue])
            parts.append("also " + _messy(rng.choice(PHRASES[other]), rng))
        parts.append(rng.choice(FILLER))
        label = rng.choice(QUEUES) if rng.random() < label_noise else queue
        rows.append({"text": ", ".join(p for p in parts if p), "queue": label, "true_queue": queue})
    return pd.DataFrame(rows)


# Hand-written rules, checked in order; the first match wins.
RULES = [
    ("shop_floor", ["hmi", "plc", "line", "operator", "machine", "historian", "scanner"]),
    ("access", ["password", "account", "access", "mfa", "permission", "locked"]),
    ("network", ["vpn", "wifi", "internet", "network", "connect"]),
    ("endpoint", ["laptop", "pc", "screen", "keyboard", "mouse", "monitor"]),
    ("application", ["app", "erp", "email", "error", "report"]),
]


def keyword_route(text: str) -> str:
    """Return the first queue whose keyword appears; fall back to a default queue."""
    for queue, words in RULES:
        if any(w in text for w in words):
            return queue
    return "access"


def train_model(train: pd.DataFrame):
    """Word and word-pair TF-IDF features into a multinomial logistic regression."""
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=1000, C=5.0),
    )
    model.fit(train["text"], train["queue"])
    return model


def auto_route_split(model, texts: pd.Series, y_true: pd.Series, threshold: float) -> dict:
    """Auto-route only tickets where the top class probability >= threshold.
    Report how many we could automate and how accurate those automated ones are."""
    proba = model.predict_proba(texts)
    confident = proba.max(axis=1) >= threshold
    pred = model.classes_[proba.argmax(axis=1)]
    return {
        "auto_routed_share": round(confident.mean(), 2),
        "auto_routed_accuracy": round(accuracy_score(y_true[confident], pred[confident]), 3),
    }


def evaluate(seed: int = 42):
    data = make_tickets(seed=seed)
    train, test = train_test_split(data, test_size=0.3, random_state=seed, stratify=data["queue"])

    kw_pred = test["text"].map(keyword_route)
    model = train_model(train)
    ml_pred = model.predict(test["text"])

    results = pd.DataFrame({
        "accuracy": [accuracy_score(test["queue"], kw_pred), accuracy_score(test["queue"], ml_pred)],
        "macro_f1": [f1_score(test["queue"], kw_pred, average="macro"),
                     f1_score(test["queue"], ml_pred, average="macro")],
    }, index=["keyword rules", "tfidf + logreg"]).round(3)

    thresholds = pd.DataFrame(
        {t: auto_route_split(model, test["text"], test["queue"].values, t) for t in (0.5, 0.7, 0.9)}
    ).T
    thresholds.index.name = "threshold"
    # Best possible score: a perfect router still "misses" the mislabeled test tickets
    ceiling = round((test["queue"] == test["true_queue"]).mean(), 3)
    return results, thresholds, ceiling, len(train), len(test)


if __name__ == "__main__":
    results, thresholds, ceiling, n_train, n_test = evaluate()
    print(f"train={n_train} test={n_test} label ceiling={ceiling}\n")
    print(results.to_string(), "\n")
    print("Auto-routing by model confidence:")
    print(thresholds.to_string())

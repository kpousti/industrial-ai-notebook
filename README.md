# Industrial AI Notebook

![tests](https://github.com/kpousti/industrial-ai-notebook/actions/workflows/tests.yml/badge.svg)

Small, working prototypes of how AI and automation can fix real problems in industrial IT: equipment monitoring, ticket triage, asset tracking, document processing and more.

Each case study follows the same format: **problem → inputs → approach → results → takeaway**, with runnable code and tests.

> All data is synthetic. Nothing here comes from, or describes, any specific company's systems.

## Case studies

| # | Topic | Techniques | Status |
|---|---|---|---|
| [001](case-studies/001-telemetry-anomaly-detection) | Catching failing equipment early from sensor telemetry | Rolling z-score, Isolation Forest | ✅ |

## Run any case study

```bash
pip install -r requirements.txt
cd case-studies/001-telemetry-anomaly-detection
python detect.py
python -m pytest
```

## About

Built by [Kia Pousti](https://github.com/kpousti), a software developer in industrial IT in San Diego exploring applied AI.

# 001 · Catching failing equipment early from sensor telemetry

## Problem
Industrial equipment streams sensor data around the clock, but alerts are often simple fixed thresholds. They miss slow degradation and fire on harmless noise, so people learn to ignore them.

## Inputs
Hourly telemetry from rotating equipment: vibration, bearing temperature, exhaust temperature. **All synthetic**, generated in `detect.py` with two injected fault types:
- **Spikes:** sudden vibration jumps (e.g. a loose component)
- **Drift:** bearing temperature and vibration creeping up together over ~2 days

## Approach
| Detector | Idea | Strength |
|---|---|---|
| Rolling z-score | Flag any sensor > 3.5σ from its trailing 48-hour mean | Simple, explainable per sensor |
| Isolation Forest | Learn "normal" from a healthy window, flag rows that are easy to isolate | Sees combinations across sensors |

## Results
| Detector | Precision | Recall | Alerts |
|---|---|---|---|
| Rolling z-score | 1.00 | 0.22 | 11 |
| Isolation Forest | 0.65 | 0.47 | 37 |

## Takeaway
The baseline never cried wolf but caught under a quarter of faults: it nails sudden spikes and is blind to slow drift. Isolation Forest caught about twice as many, at the cost of some false alarms. In practice you'd run both: the z-score as a high-confidence page, the forest as a "look at this today" queue.

## Next
- Add a drift-specific detector (CUSUM) and compare on the slow fault alone
- Measure *time to detection*, not just recall: catching drift 6 hours earlier matters more than catching it at all

## Run
```bash
pip install -r ../../requirements.txt
python detect.py
python -m pytest
```

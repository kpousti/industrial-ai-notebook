# 002 · Routing IT tickets to the right support queue

## Problem
Every IT ticket lands in one inbox and someone has to read it and forward it to the right team. When that dispatcher guesses wrong, the ticket bounces between queues and the person on the shop floor waits longer. Keyword rules help, but users describe problems loosely and mention side symptoms, so rules misfire.

## Inputs
1,500 short ticket descriptions across five queues: `network`, `access`, `endpoint`, `application`, `shop_floor`. **All synthetic**, generated in `triage.py` to look like real tickets:
- 1-2 problem phrases, with words randomly dropped and occasional typos (`eamil`, `readre`)
- 35% mention a side symptom from another queue (`..., also laptop battery drains fast`)
- ~5% of labels are deliberately wrong, as in real history where tickets get closed in the wrong queue

70% is used for training, 30% for testing. Because of the wrong labels, even a perfect router would only score **0.94** on the test set.

## Approach
| Router | Idea | Strength |
|---|---|---|
| Keyword rules | Ordered list of "if the text contains X, send to queue Y" | No training, easy to explain |
| TF-IDF + logistic regression | Weight words and word pairs by how distinctive they are, learn which ones point to each queue | Weighs all the evidence instead of stopping at the first keyword |

On top of the model, a **confidence threshold** decides what to automate: tickets where the model's top probability is above the threshold are routed automatically, the rest go to a human.

## Results
| Router | Accuracy | Macro F1 |
|---|---|---|
| Keyword rules | 0.664 | 0.661 |
| TF-IDF + logistic regression | 0.858 | 0.847 |

| Confidence threshold | Share auto-routed | Accuracy on those |
|---|---|---|
| 0.5 | 93% | 0.876 |
| 0.7 | 79% | 0.919 |
| 0.9 | 46% | 0.922 |

## Takeaway
The learned router cut misroutes by more than half versus keyword rules (from about 1 in 3 to about 1 in 7), mostly because rules stop at the first keyword they see while the model weighs everything in the ticket. The bigger practical win is the threshold: you don't have to automate everything. At 0.7 the model handles about four in five tickets at roughly 92% accuracy and hands the unclear ones to a person. Past 0.7 you give up a lot of automation for almost no extra accuracy, because the remaining errors are mostly the wrong labels in the history, not model mistakes.

## Next
- Clean the training labels (e.g. flag tickets that were reassigned) and see how much the ceiling moves
- Add the ticket's site or asset type as a feature: a "screen frozen" on a production line is almost always `shop_floor`

## Run
```bash
pip install -r ../../requirements.txt
python triage.py
python -m pytest
```

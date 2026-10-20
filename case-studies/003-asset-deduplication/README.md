# 003 · Finding duplicate devices across asset inventories

## Problem
The same machine is listed in several systems: an asset register typed by hand, a network scan, and a purchasing list. Each one writes it differently (`NOR-WS-0012`, `nor-ws-0012.plant.local`, a serial with a typo, a MAC with dashes). Without merging them you can't answer basic questions like "how many workstations do we have?" or "which devices have no owner?"

## Inputs
749 records describing 360 devices across 6 sites. **All synthetic**, generated in `dedupe.py`. Each device appears in 1-3 sources, and each source has its own habits:
- **Register:** hand-typed, so it has hostname and serial typos, sometimes lower case, and a MAC only half the time
- **Network scan:** full hostname with a domain suffix and a dashed MAC, but no serial
- **Purchasing:** serial from the invoice, a hostname only 40% of the time, and no MAC

Hostnames are sequential per site, so two *different* devices often look almost the same (`-0012` vs `-0013`). That's the hard part.

## Approach
1. **Blocking.** Only compare records at the same site with the same model. That cuts 280,126 possible pairs to 5,833 (48x fewer) without losing any true duplicate, since a device never changes site or model.
2. **Pair features.** Hostname similarity and exact match (after removing case and domain), serial similarity, and MAC match (after removing formatting). Each feature also gets an "is missing" flag, so "no serial to compare" isn't treated like "serials disagree".
3. **Matchers.**

| Matcher | Idea |
|---|---|
| Exact-key rule | Merge only if serial or MAC match exactly |
| Similarity + logistic regression | Learn how much each feature counts, trained on 3 sites and tested on the other 3 |

4. **Grouping.** Matched pairs are joined with union-find: if A matches B and B matches C, all three are one device, even if A and C share no field.

## Results
On the 3 test sites (180 true devices):

| Matcher | Pair precision | Pair recall | Pair F1 | Devices found | Devices exactly right |
|---|---|---|---|---|---|
| Exact-key rule | 1.000 | 0.454 | 0.625 | 258 | 58% |
| Similarity + logistic regression | 1.000 | 0.765 | 0.867 | 204 | 87% |

Neither matcher merged two different devices. The learned matcher rebuilt 87% of devices exactly, versus 58% for the rule.

## Takeaway
Most of the gain comes from cleaning fields before comparing them, not from fuzzy matching. The model gave a large weight to *exact* hostname match after normalizing case and domain, and a *negative* weight to fuzzy hostname similarity. That's because a hostname with a typo looks just as similar as the next machine in the sequence. Grouping also matters: a network-scan record and a purchasing record often share no field at all, and they only come together through the register. About 19% of true pairs share no usable field, so no pairwise matcher can find them directly.

One modeling lesson: with the default regularization (C=1), the model shrank its weights so much that it missed some exact serial matches the simple rule caught. Lighter regularization (C=100) fixed that. It's worth checking that a model never does worse than the rule it's replacing.

## Next
- Use the one-record-per-source constraint: a device can't have two network-scan records, which would split some wrong merges
- Send the leftover uncertain pairs to a person, with both records side by side

## Run
```bash
pip install -r ../../requirements.txt
python dedupe.py
python -m pytest
```

<div align="center">

# Bot / Not

### Telling automated accounts apart from real people — by how they behave, not just what they post.

[**▶ Live showcase**](https://enkai-liu.github.io/Bot-or-Not-/) &nbsp;·&nbsp; 22 behavioural signals &nbsp;·&nbsp; 889 accounts &nbsp;·&nbsp; 24,769 posts

![Python](https://img.shields.io/badge/Python-3.10+-0a0b0e?style=flat-square&logo=python&logoColor=d2f24a)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-0a0b0e?style=flat-square&logo=scikitlearn&logoColor=d2f24a)
![accuracy](https://img.shields.io/badge/accuracy-96%25-d2f24a?style=flat-square)
![bot recall](https://img.shields.io/badge/bot_recall-0.86-d2f24a?style=flat-square)

</div>

---

## What it does

Given a stream of social-media posts and bare-bones profile data, **Bot / Not** flags
the accounts that aren't run by humans. The hard part isn't accuracy — bots are the
minority, so a model that calls *everything* human already scores ~80%. The hard part is
**recall**: actually catching the bots. This pipeline reduces every account to a
behavioural fingerprint (posting rhythm, off-hours bursts, text shape, entity use) and
trains a calibrated, cross-validated classifier on top of it.

> The original version of this project was rushed. This is the rebuild — same problem,
> done properly. The headline change: **bot recall went from 0.48 to 0.86.**

## Results

All numbers are **out-of-fold** from stratified 5-fold cross-validation — every account is
scored by a model that never trained on it.

| Metric | Original | **Rebuilt** | Δ |
|---|---:|---:|---:|
| Accuracy | 0.91 | **0.96** | +0.05 |
| Bot recall | 0.48 | **0.86** | **+0.38** |
| Bot F1 | 0.63 | **0.89** | +0.26 |
| ROC-AUC | — | **0.98** | — |

What changed:

- **Used all the data.** The first version trained on 2 of 4 dataset rounds; this uses all 889 users.
- **Engineered behaviour.** 22 signals — inter-post timing variability, off-hours posting, length variability, link/hashtag ratios — instead of raw averages.
- **Fixed the imbalance.** `class_weight="balanced"` so the model is rewarded for catching the rare bots, not for playing it safe.
- **Benchmarked honestly.** Four model families compared under cross-validation; the winner chosen on evidence (see below).

### Model bake-off

| Model | ROC-AUC | PR-AUC | Bot recall | Bot F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.894 | 0.769 | 0.755 | 0.647 |
| Random Forest (default) | 0.986 | 0.966 | 0.777 | 0.864 |
| **Random Forest (balanced)** ◆ | 0.983 | 0.959 | **0.864** | 0.893 |
| Gradient Boosting | 0.985 | 0.967 | 0.853 | 0.902 |

The balanced Random Forest is selected: it recovers the recall the original lost, stays
interpretable (clean feature importances), and trails the gradient-boosted model by a
hair on F1 while beating it on recall — the metric that matters here.

### What gives a bot away

Top signals by importance: **hashtag usage**, **post-length variability**,
**off-hours posting**, and **posting-rhythm regularity**. Humans are irregular; bots are
not. The full ranking is in the [live showcase](https://enkai-liu.github.io/Bot-or-Not-/).

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py          # parse → engineer → cross-validate → report → export
python -m pytest        # sanity tests for parsing + feature extraction
```

`python main.py` prints the full report and regenerates `data/features.csv` plus the
data the web showcase reads (`docs/data/results.{json,js}`). Flags: `--no-export`,
`--quiet`.

## How it works

```
datasets/*.json ─▶ parser.py ─▶ features.py ─▶ model.py ─▶ export.py ─▶ docs/
  raw users &      join posts    22 signals    5-fold CV,   metrics +     interactive
  posts + labels   to authors    per account   model        per-user      showcase
                   + bot labels                 bake-off     predictions
```

| File | Role |
|---|---|
| [`src/parser.py`](src/parser.py) | Load raw JSON + bot-label files into `User` objects. Labels come *only* from the label files — no hard-coded UUIDs. |
| [`src/features.py`](src/features.py) | Reduce a user's posts + profile to 22 numeric behavioural signals. |
| [`src/dataset.py`](src/dataset.py) | Assemble the user-level feature table across all rounds. |
| [`src/model.py`](src/model.py) | Cross-validate, benchmark four models, rank feature importances. |
| [`src/export.py`](src/export.py) | Serialize metrics, curves, and per-user verdicts for the frontend. |
| [`main.py`](main.py) | One-command end-to-end pipeline. |
| [`docs/`](docs/) | Zero-build static showcase (vanilla JS, hand-rolled SVG charts). |

## The showcase

[`docs/`](docs/) is a dependency-free static site that visualises the *real* model output:
an animated population scan, the before/after story, the model bake-off, ROC/PR curves, a
confusion matrix, ranked feature importances, and an **interactive explorer** that places
all 889 accounts by their predicted bot probability — click any one to open its dossier.

To publish: enable **GitHub Pages → Deploy from branch → `main` / `docs`**. The page works
offline too (data is embedded as JS), so you can just open `docs/index.html`.

## Data

Four rounds of labelled social-media activity (`datasets/`): **889 users**, **24,769
posts**, **184 bots** (~21%). Each round ships a `dataset.posts&users.NN.json` and a
`dataset.bots.NN.txt` of ground-truth bot UUIDs.

## Tech

Python · pandas · NumPy · scikit-learn · python-dateutil · vanilla JS + SVG (no frontend build step).

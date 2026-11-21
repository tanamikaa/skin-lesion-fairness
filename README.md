# P01 — Fair & Explainable Skin-Lesion Classification: CNN vs Vision Transformer

**Team 1** · Afzal Raza · Anamika Kumari · Khushi Narwariya · Karanpal

## Problem statement
Classify dermoscopic images into the 7 HAM10000 lesion types and study whether accuracy
drops on darker skin tones. Compare a CNN (ResNet-50) against a Vision Transformer (ViT-B/16)
on two axes:

1. **Fairness** — per-Fitzpatrick-skin-type performance gap.
2. **Explainability** — quality of Grad-CAM (CNN) vs attention maps (ViT).

**Research question:** *Is ViT or CNN fairer across skin tones, and which one gives better explanations?*

## Datasets
| Dataset | Use | Source |
|---|---|---|
| HAM10000 | Train / val / test (7 classes) | Harvard Dataverse — https://doi.org/10.7910/DVN/DBW86T |
| Fitzpatrick17k | Skin-tone (FST I–VI) evaluation | https://github.com/mattgroh/fitzpatrick17k |

Data is **never committed** to git. See [data/README.md](data/README.md) for the expected layout.

> ⚠️ Fitzpatrick17k uses 114 clinical conditions and *clinical* (not dermoscopic) photos.
> We must map its labels onto the 7 HAM10000 classes and report the domain shift honestly.
> See [docs/TEAM_ROLES.md](docs/TEAM_ROLES.md) → Afzal's tasks.

## Approach (baseline → advanced)
1. ResNet-50 transfer learning baseline
2. Fine-tuned ViT-B/16
3. Class-imbalance handling: weighted cross-entropy → focal loss, oversampling
4. Explainability: Grad-CAM (CNN), attention rollout (ViT), explanation-quality metrics
5. Fairness: metrics per Fitzpatrick type, gap / ratio between lightest and darkest groups

## Repository layout
```
configs/            YAML experiment configs (one per model)
data/               local data only (git-ignored) + README with layout
docs/               team roles, timeline, meeting notes
notebooks/          exploration only — real code goes in src/
reports/            final report, slides, figures
results/            metrics JSON / CSV (small files only), checkpoints git-ignored
scripts/            CLI entry points (train, evaluate, explain, compare_models, hparam_search, download_fitzpatrick)
src/skinfair/
  data/             datasets, splits, transforms, sampling        — Afzal
  models/           resnet.py — Anamika, vit.py — Khushi
  losses.py         focal / weighted CE                           — Anamika
  train.py          shared training loop                          — Anamika + Khushi
  evaluation/       metrics + fairness analysis                   — Karanpal
  explain/          gradcam — Anamika, attention — Khushi, quality — Karanpal
tests/              pytest unit tests
```

## Quick start
```bash
git clone <repo-url> && cd skin-lesion-fairness
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

pytest                                           # unit tests (synthetic data, no download needed)

# train (override any config value with --set, e.g. loss / oversampling experiments)
python scripts/train.py --config configs/resnet50.yaml
python scripts/train.py --config configs/vit_b16.yaml --set loss.name=weighted_ce data.oversample=true

# evaluate: predictions CSVs + metrics.json (overall, per skin type, gaps, bootstrap CIs)
python scripts/evaluate.py --config configs/resnet50.yaml --checkpoint results/checkpoints/resnet50_best.pt

# explanations + deletion/insertion quality, then CNN-vs-ViT fairness comparison
python scripts/explain.py  --config configs/resnet50.yaml --checkpoint results/checkpoints/resnet50_best.pt --split fitz
python scripts/compare_models.py --a results/resnet50/predictions_fitz.csv --b results/vit_b16/predictions_fitz.csv --name-a ResNet-50 --name-b ViT-B/16

# equal-budget hyper-parameter search for both models
python scripts/hparam_search.py --trials 8 --epochs 10
```

## Team & work split
Full breakdown with tasks and deliverables: **[docs/TEAM_ROLES.md](docs/TEAM_ROLES.md)**

| Member | Area |
|---|---|
| Afzal Raza | Data pipeline, splits, imbalance sampling, Fitzpatrick label mapping, EDA |
| Anamika Kumari | ResNet-50 baseline, focal loss, training loop, Grad-CAM |
| Khushi Narwariya | ViT-B/16 fine-tuning, attention rollout, hyper-parameter search |
| Karanpal | Evaluation & fairness metrics, explanation-quality metrics, statistical tests |

How we work together (branches, PRs, reviews): **[CONTRIBUTING.md](CONTRIBUTING.md)**

## References
1. Esteva et al., *Dermatologist-level classification of skin cancer with deep neural networks*, Nature 2017 — https://doi.org/10.1038/nature21056
2. Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR 2021 — https://arxiv.org/abs/2010.11929
3. Groh et al., *Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset*, CVPR-W 2021 — https://arxiv.org/abs/2104.09957
4. Tschandl et al., *The HAM10000 dataset*, Scientific Data 2018 — https://doi.org/10.1038/sdata.2018.161

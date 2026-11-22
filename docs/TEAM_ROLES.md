# Team Roles & Work Distribution

Each member owns one **technical module** end-to-end (code + tests + the matching report
section) and is the **reviewer** for one other module. Work is split so every person has
roughly the same load: one core pipeline component, one explainability/analysis piece,
and one report section.

| Member | Owns | Reviews | Report section |
|---|---|---|---|
| **Afzal Raza** | `src/skinfair/data/`, EDA notebook | Karanpal's evaluation code | Datasets & preprocessing |
| **Anamika Kumari** | `models/resnet.py`, `losses.py`, `train.py`, `explain/gradcam.py` | Khushi's ViT code | CNN baseline + methods |
| **Khushi Narwariya** | `models/vit.py`, `explain/attention.py`, HP search | Anamika's CNN code | ViT + attention results |
| **Karanpal** | `evaluation/`, `explain/quality.py`, stats tests | Afzal's data code | Fairness analysis + discussion |

Each member plans their own work. The intro, conclusion and slides are written together.

---

## 1. Afzal Raza — Data pipeline
- [ ] Download HAM10000 + Fitzpatrick17k, document exact steps in `data/README.md`
- [x] `datasets.py`: PyTorch `Dataset` classes for both datasets (image, label, `fitzpatrick` / `None`)
- [x] `splits.py`: **lesion-level** stratified train/val/test split for HAM10000 (same `lesion_id` must never appear in two splits — avoids leakage)
- [x] Map Fitzpatrick17k labels → the 7 HAM classes (`mel, nv, bcc, akiec, bkl, df, vasc`); drop / document unmapped ones
- [x] `transforms.py`: train augmentations + eval transforms (224×224, ImageNet normalisation)
- [x] `sampling.py`: `WeightedRandomSampler` for oversampling minority classes
- [x] EDA notebook: class distribution, Fitzpatrick distribution, sample grids
- **Deliverable:** `get_dataloaders(config)` that the training code calls

## 2. Anamika Kumari — CNN baseline + Grad-CAM
- [x] `models/resnet.py`: ImageNet-pretrained ResNet-50, new 7-class head, freeze/unfreeze option
- [x] `losses.py`: weighted cross-entropy + focal loss (γ, α configurable)
- [x] `train.py`: shared training loop (AMP, early stopping, checkpointing, logging) — used by both models
- [ ] Experiments: CE vs weighted CE vs focal loss; with/without oversampling
- [x] `explain/gradcam.py`: Grad-CAM on the last conv block
- **Deliverable:** best ResNet-50 checkpoint + Grad-CAM maps for the test set

## 3. Khushi Narwariya — Vision Transformer + attention maps
- [x] `models/vit.py`: ViT-B/16 via `timm`, 7-class head, layer-wise LR decay
- [x] Fine-tuning recipe (warm-up, cosine LR, stronger augmentation / mixup if helpful)
- [x] Same loss/sampling variants as the CNN so the comparison is fair (`--set loss.name=... data.oversample=...`)
- [~] Hyper-parameter search for **both** models with equal budget — code ready (`scripts/hparam_search.py`), needs a real run
- [x] `explain/attention.py`: attention rollout (+ optionally Grad-CAM for ViT for a like-for-like comparison)
- **Deliverable:** best ViT checkpoint + attention maps for the test set

## 4. Karanpal — Fairness & explanation-quality evaluation
- [x] `evaluation/metrics.py`: accuracy, balanced accuracy, macro-F1, per-class recall, AUROC, confusion matrix
- [x] `evaluation/fairness.py`: metrics **per Fitzpatrick type (I–VI)**, gap (max − min), ratio (min / max), light (I–III) vs dark (IV–VI)
- [x] Bootstrap 95% confidence intervals + significance test for CNN vs ViT gap
- [x] `explain/quality.py`: deletion / insertion AUC, and (if masks available) pointing-game / IoU with lesion area
- [ ] Results tables + plots in `reports/figures/`
- **Deliverable:** `scripts/evaluate.py` that outputs one JSON with all metrics for a checkpoint

---

## Interfaces (agree on these first so everyone can work in parallel)
- Dataloader batch = `dict(image: Tensor[B,3,224,224], label: LongTensor[B], fitzpatrick: LongTensor[B] (−1 if unknown), image_id: list[str])`
- Model: `build_model(cfg) -> nn.Module` returning logits `[B, 7]`
- Predictions file (written by `evaluate.py`): CSV with `image_id, label, pred, prob_0..prob_6, fitzpatrick`
- Explanation: `explain(model, images) -> Tensor[B, 224, 224]` heat-maps in `[0, 1]`


---

## Status
Done (code, tests, and one real run per model on Colab T4, focal loss, seed 42; see `results/EXPERIMENTS.md`):
- Data pipeline, Fitzpatrick17k mapping (1,916 images), EDA notebook and figures
- ResNet-50 + ViT-B/16 training, Grad-CAM, attention rollout, evaluation, fairness comparison, explanation-quality scores
- `reports/REPORT.md`: methods, results, discussion, and draft introduction / conclusion

Still open:
- CE / weighted-CE vs focal, with / without oversampling (code ready, not run); hyper-parameter search; more seeds
- Team review of `reports/REPORT.md` (intro + conclusion are drafts), slides, final submission
- Lesion masks for pointing-game / IoU are supported by `explain/quality.py` but not wired into `scripts/explain.py`
- HAM10000 split depends on the scikit-learn version (Colab test n = 1,441, local 1,431); save the split to CSV to make it exact

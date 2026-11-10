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
- [ ] `datasets.py`: PyTorch `Dataset` classes for both datasets (image, label, `fitzpatrick` / `None`)
- [ ] `splits.py`: **lesion-level** stratified train/val/test split for HAM10000 (same `lesion_id` must never appear in two splits — avoids leakage)
- [ ] Map Fitzpatrick17k labels → the 7 HAM classes (`mel, nv, bcc, akiec, bkl, df, vasc`); drop / document unmapped ones
- [ ] `transforms.py`: train augmentations + eval transforms (224×224, ImageNet normalisation)
- [ ] `sampling.py`: `WeightedRandomSampler` for oversampling minority classes
- [ ] EDA notebook: class distribution, Fitzpatrick distribution, sample grids
- **Deliverable:** `get_dataloaders(config)` that the training code calls

## 2. Anamika Kumari — CNN baseline + Grad-CAM
- [ ] `models/resnet.py`: ImageNet-pretrained ResNet-50, new 7-class head, freeze/unfreeze option
- [ ] `losses.py`: weighted cross-entropy + focal loss (γ, α configurable)
- [ ] `train.py`: shared training loop (AMP, early stopping, checkpointing, logging) — used by both models
- [ ] Experiments: CE vs weighted CE vs focal loss; with/without oversampling
- [ ] `explain/gradcam.py`: Grad-CAM on the last conv block
- **Deliverable:** best ResNet-50 checkpoint + Grad-CAM maps for the test set

## 3. Khushi Narwariya — Vision Transformer + attention maps
- [ ] `models/vit.py`: ViT-B/16 via `timm`, 7-class head, layer-wise LR decay
- [ ] Fine-tuning recipe (warm-up, cosine LR, stronger augmentation / mixup if helpful)
- [ ] Same loss/sampling variants as the CNN so the comparison is fair
- [ ] Hyper-parameter search (LR, batch size, epochs) for **both** models with equal budget
- [ ] `explain/attention.py`: attention rollout (+ optionally Grad-CAM for ViT for a like-for-like comparison)
- **Deliverable:** best ViT checkpoint + attention maps for the test set

## 4. Karanpal — Fairness & explanation-quality evaluation
- [ ] `evaluation/metrics.py`: accuracy, balanced accuracy, macro-F1, per-class recall, AUROC, confusion matrix
- [ ] `evaluation/fairness.py`: metrics **per Fitzpatrick type (I–VI)**, gap (max − min), ratio (min / max), light (I–III) vs dark (IV–VI)
- [ ] Bootstrap 95% confidence intervals + significance test for CNN vs ViT gap
- [ ] `explain/quality.py`: deletion / insertion AUC, and (if masks available) pointing-game / IoU with lesion area
- [ ] Results tables + plots in `reports/figures/`
- **Deliverable:** `scripts/evaluate.py` that outputs one JSON with all metrics for a checkpoint

---

## Interfaces (agree on these first so everyone can work in parallel)
- Dataloader batch = `dict(image: Tensor[B,3,224,224], label: LongTensor[B], fitzpatrick: LongTensor[B] (−1 if unknown), image_id: list[str])`
- Model: `build_model(cfg) -> nn.Module` returning logits `[B, 7]`
- Predictions file (written by `evaluate.py`): CSV with `image_id, label, pred, prob_0..prob_6, fitzpatrick`
- Explanation: `explain(model, images) -> Tensor[B, 224, 224]` heat-maps in `[0, 1]`

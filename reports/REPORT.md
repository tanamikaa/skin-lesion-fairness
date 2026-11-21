# Fair & Explainable Skin-Lesion Classification: CNN vs Vision Transformer

> Draft skeleton. Method sections describe what the code does; every **TBD** needs real results
> from runs on HAM10000 / Fitzpatrick17k. Do not fill a TBD with numbers that were not produced by
> `scripts/evaluate.py`, `scripts/compare_models.py` or `scripts/explain.py`.

## 1. Introduction  *(written together)*
TBD

## 2. Datasets & preprocessing  *(Afzal)*
- **HAM10000**: 7 dermoscopic classes, strongly imbalanced (nv dominates). Split **at lesion level**
  (`StratifiedGroupKFold`, grouped by `lesion_id`, stratified by `dx`, ~70/15/15) so images of one lesion never
  appear in two splits; `assert_no_leakage` enforces it.
- **Fitzpatrick17k**: clinical photos with Fitzpatrick skin type (I-VI). Only conditions mappable to the
  7 HAM classes are kept (`fitzpatrick_mapping.py`). Deliberately unmapped: invasive squamous cell carcinoma
  (not HAM `akiec`), port-wine stain (not `vasc`). Rows/class and rows/skin type: TBD (from the EDA notebook).
- **Domain shift** (dermoscopic -> clinical) is a confound: Fitzpatrick17k numbers measure robustness across
  skin tones *and* domain shift together; report it as such.
- Transforms: 224x224, ImageNet normalisation; train: random resized crop, flips, rotation, brightness/contrast/
  saturation jitter. **Hue jitter is off** so augmentation does not alter apparent skin tone.
- Class imbalance: `WeightedRandomSampler` (1/class-count weights).

## 3. Methods
### 3.1 CNN baseline  *(Anamika)*
ImageNet-pretrained ResNet-50, new 7-way head, backbone frozen for the first epochs. Losses: cross-entropy,
class-weighted CE, focal loss (gamma = 2, optional class weights). Shared training loop: AdamW, cosine schedule,
AMP on CUDA, gradient clipping, early stopping on validation **balanced accuracy**. Grad-CAM on `layer4[-1]`.
Experiment grid (CE / weighted CE / focal x with/without oversampling): TBD.

### 3.2 Vision Transformer  *(Khushi)*
timm ViT-B/16 (ImageNet-pretrained), layer-wise LR decay 0.75, 3 warm-up epochs + cosine schedule, weight decay 0.05,
optional mixup. Same losses/sampling variants and the same seeds as the CNN. Hyper-parameters chosen by an
equal-budget random search on validation balanced accuracy (`scripts/hparam_search.py`): TBD.
Explanations: attention rollout (discard ratio 0.9, CLS row -> 14x14 -> 224x224) and Grad-CAM on the last block.

### 3.3 Fairness & explanation-quality evaluation  *(Karanpal)*
- Metrics: accuracy, balanced accuracy, macro-F1, per-class recall, macro one-vs-rest AUROC, confusion matrix.
- Fairness: metrics per Fitzpatrick type, gap (max - min), ratio (min / max), light (I-III) vs dark (IV-VI)
  with **bootstrap 95% CIs** (groups resampled independently).
- CNN vs ViT: paired bootstrap on the light-minus-dark gap difference (same images, resampled jointly) and
  exact McNemar test on accuracy.
- Explanation quality: deletion AUC (lower = better), insertion AUC (higher = better), per skin-tone group;
  pointing-game / IoU implemented but need lesion masks (not wired in).

## 4. Results
| | ResNet-50 | ViT-B/16 |
|---|---|---|
| HAM10000 test balanced acc | TBD | TBD |
| Fitzpatrick17k light-minus-dark gap [95% CI] | TBD | TBD |
| Deletion AUC / Insertion AUC | TBD | TBD |

Figures go to `reports/figures/`.

## 5. Discussion & limitations  *(Karanpal)*
Small FST groups (see `n` and the CI widths), domain shift, label-mapping choices, single dataset split / seed: TBD.

## 6. Conclusion  *(written together)*
TBD

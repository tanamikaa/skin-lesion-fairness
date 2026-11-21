# Fair & Explainable Skin-Lesion Classification: CNN vs Vision Transformer

> Draft. Numbers below come from one run per model (seed 42) logged in `results/EXPERIMENTS.md`
> (`results/*/*/metrics.json`, `reports/figures/fairness_comparison.json`). Remaining **TBD**s are for the
> team: introduction, conclusion, and experiments we did not run.

## 1. Introduction  *(written together)*
TBD

## 2. Datasets & preprocessing  *(Afzal)*
- **HAM10000**: 7 dermoscopic classes, strongly imbalanced (nv dominates). Split **at lesion level**
  (`StratifiedGroupKFold`, grouped by `lesion_id`, stratified by `dx`, ~70/15/15) so images of one lesion never
  appear in two splits; `assert_no_leakage` enforces it.
  10,015 images from 7,470 lesions (1,956 lesions have several images). Class counts: nv 6,705, mel 1,113, bkl 1,099, bcc 514,
  akiec 327, vasc 142, df 115 (imbalance ratio nv/df = 58). Local split: train 7,153 / val 1,431 / test 1,431 (`reports/figures/eda_ham_classes.png`). The Colab runs evaluated on 1,441 test images: the library version of `StratifiedGroupKFold` differed, so the split is not identical across environments (leakage-free in both; see Limitations).
- **Fitzpatrick17k**: clinical photos with Fitzpatrick skin type (I-VI). Only conditions mappable to the
  7 HAM classes are kept (`fitzpatrick_mapping.py`). Deliberately unmapped: invasive squamous cell carcinoma
  (not HAM `akiec`), port-wine stain (not `vasc`). Mapped images: 1,916; per class: akiec 175, bcc 596, bkl 69, df 79, mel 573, nv 236, vasc 188. Per skin type: I 450, II 672, III 380, IV 221, V 103, VI 34, unknown 56.
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
Experiment grid: **only focal loss (class-weighted, no oversampling) was run**; the CE / weighted-CE / oversampling
comparison was not run (TBD if time allows). When oversampling is on, class weights are switched off to avoid double correction.

### 3.2 Vision Transformer  *(Khushi)*
timm ViT-B/16 (ImageNet-pretrained), layer-wise LR decay 0.75, 3 warm-up epochs + cosine schedule, weight decay 0.05,
optional mixup. Same losses/sampling variants and the same seeds as the CNN. Hyper-parameters chosen by an
equal-budget random search on validation balanced accuracy (`scripts/hparam_search.py`). **The search was not run;** both models use the config defaults (ResNet lr 1e-4, ViT lr 3e-5, batch 32, max 15 epochs, early-stopping patience 5, same seed 42).
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
Training on a Colab T4, selection on validation balanced accuracy only. Test set = HAM10000 lesion-level hold-out (n = 1,441).
Fitzpatrick17k is a **different domain** (clinical photos), n = 1,916 mappable images, used only for evaluation.

| | ResNet-50 | ViT-B/16 |
|---|---|---|
| Best val balanced acc (epoch) | 0.789 (8) | 0.821 (7) |
| HAM10000 test: accuracy / balanced acc / macro-F1 / macro-AUROC | 0.809 / 0.726 / 0.659 / 0.954 | 0.797 / 0.743 / 0.670 / 0.964 |
| Fitzpatrick17k: accuracy / balanced acc | 0.245 / 0.269 | 0.385 / 0.336 |
| Light (I-III) minus dark (IV-VI) balanced-acc gap, 95% bootstrap CI | 0.077 [0.022, 0.131] | 0.044 [-0.016, 0.102] |
| Max - min over skin types (ratio min/max) | 0.156 (0.55) | 0.121 (0.70) |

Balanced accuracy per Fitzpatrick type (n in brackets: I 450, II 672, III 380, IV 221, V 103, VI 34):

| FST | I | II | III | IV | V | VI |
|---|---|---|---|---|---|---|
| ResNet-50 | 0.350 | 0.276 | 0.193 | 0.208 | 0.202 | 0.203 |
| ViT-B/16 | 0.401 | 0.337 | 0.291 | 0.337 | 0.280 | 0.286 |

CIs for types V and VI are wide (e.g. ResNet VI: 0.113-0.389), see `reports/figures/fst_comparison.png`.

**CNN vs ViT fairness gap** (paired bootstrap on the same 1,860 images with a known skin type): difference in gap
(ResNet - ViT) = 0.034, 95% CI [-0.041, 0.109], p = 0.36. McNemar test on accuracy: ViT is right where ResNet is wrong
on 479 images, the reverse on 212 (p < 0.001), i.e. ViT is more accurate on Fitzpatrick17k overall.

**Explanations** (deletion AUC: lower is better; insertion AUC: higher is better; first 256 Fitzpatrick17k images):

| Method | Deletion AUC | Insertion AUC |
|---|---|---|
| ResNet-50 + Grad-CAM | 0.310 | 0.544 |
| ViT-B/16 + attention rollout | 0.239 | 0.441 |
| ViT-B/16 + Grad-CAM | 0.259 | 0.453 |

Per-skin-type values are in `results/*/*/explain/fitz/quality_*.csv`; groups V and VI have only 14 and 6 images, so differences there are not interpretable.

## 5. Discussion & limitations  *(Karanpal)*
What the numbers support:
- On HAM10000 the two models are close (balanced acc 0.726 vs 0.743); ViT has the higher validation score.
- On Fitzpatrick17k both models drop sharply (balanced acc 0.27 and 0.34, chance is about 0.14). This is mostly dermoscopic -> clinical **domain shift**, not only skin tone.
- Both models score lower on dark skin. The ResNet gap (0.077) has a CI above zero; the ViT gap (0.044) has a CI that includes zero. The direct comparison is **not significant** (p = 0.36), so we cannot claim that ViT is fairer; ViT is more accurate overall on Fitzpatrick17k.
- Per-class recall by light vs dark (`per_class_recall_light_vs_dark` in `metrics.json`) shows no consistent pattern; `df` has only 8 dark-skin images and recall 0 for both models.

Limitations: the HAM10000 split depends on the scikit-learn version (Colab test n = 1,441 vs 1,431 locally), so numbers should be reproduced in a single environment; one seed and one training run per model; default hyper-parameters (no search); only the focal-loss setting; HAM10000 itself has no skin-type labels, so fairness is measured only on Fitzpatrick17k; the label mapping to 7 classes drops most conditions (1,916 of 16,577 images) and leaves squamous cell carcinoma and port-wine stain unmapped on purpose; skin types V and VI have only 103 and 34 images, and `df`/`bkl` are small. Explanation-quality numbers use 256 images (6 of type VI), and deletion/insertion AUC depends on each model's probability scale, so cross-model comparison is only indicative. Pointing-game / IoU were not computed (no lesion masks).

## 6. Conclusion  *(written together)*
TBD

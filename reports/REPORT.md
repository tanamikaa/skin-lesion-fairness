# Fair & Explainable Skin-Lesion Classification: CNN vs Vision Transformer

> Repository version of the submitted paper (`reports/paper/`). Numbers come from one run per setting (seed 42; four runs:
> ResNet-50 and ViT-B/16, each without and with oversampling), logged in `results/EXPERIMENTS.md`
> (`results/*/*/metrics.json`, `reports/figures/*/fairness_comparison.json`). The plain CE / weighted-CE comparison and the
> hyper-parameter search were not run.

## 1. Introduction
Deep-learning classifiers for skin lesions are usually trained on dermoscopic data that is dominated by lighter
skin, so a model that looks accurate overall may work worse for patients with darker skin. We study the 7-class
HAM10000 lesion classification task with two model families, a CNN (ResNet-50) and a Vision Transformer
(ViT-B/16), and ask: **which one is fairer across Fitzpatrick skin types, and which gives better explanations
(Grad-CAM for the CNN, attention rollout for the ViT)?** HAM10000 has no skin-type labels, so fairness is measured on
Fitzpatrick17k, restricted to the conditions that map onto the 7 HAM classes. Both datasets are heavily imbalanced
(nv is 67% of HAM10000), which we address with focal loss and, in a second pair of runs, class-balanced oversampling. Related work: Esteva et al. (2017) for CNN
dermatology classifiers, Dosovitskiy et al. (2021) for ViT, and Groh et al. (2021) for Fitzpatrick17k.

## 2. Datasets & preprocessing  *(Afzal)*
- **HAM10000**: 7 dermoscopic classes, strongly imbalanced (nv dominates). Split **at lesion level**
  (`StratifiedGroupKFold`, grouped by `lesion_id`, stratified by `dx`, ~70/15/15) so images of one lesion never
  appear in two splits; `assert_no_leakage` enforces it.
  10,015 images from 7,470 lesions (1,956 lesions have several images). Class counts: nv 6,705, mel 1,113, bkl 1,099, bcc 514,
  akiec 327, vasc 142, df 115 (imbalance ratio nv/df = 58). Local split: train 7,153 / val 1,431 / test 1,431 (`reports/figures/eda_ham_classes.png`). The Colab runs evaluated on 1,441 test images: the library version of `StratifiedGroupKFold` differed, so the split is not identical across environments (leakage-free in both; see Limitations).
- **Fitzpatrick17k**: clinical photos with Fitzpatrick skin type (I-VI). Only conditions mappable to the
  7 HAM classes are kept (`fitzpatrick_mapping.py`). Deliberately unmapped: invasive squamous cell carcinoma
  (not HAM `akiec`), port-wine stain (not `vasc`). Mapped images: 1,916; per class: akiec 175, bcc 596, bkl 69, df 79, mel 573, nv 236, vasc 188. Per skin type: I 450, II 672, III 380, IV 221, V 103, VI 34, unknown 56.
  Only 400 of these images could still be downloaded from the original URLs; the other 1,516 come from an unofficial
  Kaggle mirror of the original, unprocessed images (`mobaswiralfarabi/fitzpatrick17k-original`), named by the MD5 hash
  used in the official CSV. Fitzpatrick17k is licensed CC BY-NC-SA 3.0.
- **Domain shift** (dermoscopic -> clinical) is a confound: Fitzpatrick17k numbers measure robustness across
  skin tones *and* domain shift together; report it as such.
- Transforms: 224x224, ImageNet normalisation; train: random resized crop, flips, rotation, brightness/contrast/
  saturation jitter. **Hue jitter is off** so augmentation does not alter apparent skin tone.
- Class imbalance: `WeightedRandomSampler` (1/class-count weights).

## 3. Methods
### 3.1 CNN baseline  *(Anamika)*
ImageNet-pretrained ResNet-50, new 7-way head, backbone frozen for the first epochs. Losses implemented: cross-entropy,
class-weighted CE and focal loss (gamma = 2, optional class weights). Shared training loop: AdamW, cosine schedule,
AMP on CUDA, gradient clipping, early stopping on validation **balanced accuracy**. Grad-CAM on `layer4[-1]`.
Experiment grid: **focal loss only**, each model trained (a) with class-weighted focal loss and no oversampling (the baseline)
and (b) with class-balanced oversampling. The plain CE / weighted-CE comparison was not run. When oversampling is on, class
weights are switched off to avoid double correction.

### 3.2 Vision Transformer  *(Khushi)*
timm ViT-B/16 (ImageNet-pretrained), layer-wise LR decay 0.75, 3 warm-up epochs + cosine schedule, weight decay 0.05,
mixup off. Same focal-loss runs (with and without oversampling) and the same seed as the CNN. An equal-budget random
search over hyper-parameters is implemented (`scripts/hparam_search.py`) but **was not run**; both models use the config defaults (ResNet lr 1e-4, ViT lr 3e-5, batch 32, max 15 epochs, early-stopping patience 5, same seed 42).
Explanations: attention rollout (discard ratio 0.9, CLS row -> 14x14 -> 224x224) and Grad-CAM on the last block.

### 3.3 Fairness & explanation-quality evaluation  *(Karanpal)*
- Metrics: accuracy, balanced accuracy, macro-F1, per-class recall, macro one-vs-rest AUROC, confusion matrix.
- Fairness: metrics per Fitzpatrick type, gap (max - min), ratio (min / max), light (I-III) vs dark (IV-VI)
  with **bootstrap 95% CIs** (groups resampled independently, 1,000 resamples).
- CNN vs ViT: paired bootstrap on the light-minus-dark gap difference (same images, resampled jointly, 2,000 resamples) and
  exact McNemar test on accuracy.
- Explanation quality: deletion AUC (lower = better), insertion AUC (higher = better), per skin-tone group;
  pointing-game / IoU implemented but not evaluated: the HAM10000 lesion masks (a separate Dataverse download) were not used.

## 4. Results
Training on a Colab T4, selection on validation balanced accuracy only. Test set = HAM10000 lesion-level hold-out (n = 1,441).
Fitzpatrick17k is a **different domain** (clinical photos), n = 1,916 mappable images, used only for evaluation.

| | ResNet-50 | ViT-B/16 |
|---|---|---|
| Best val balanced acc (epoch) | 0.789 (8) | 0.821 (7) |
| HAM10000 test: accuracy / balanced acc / macro-F1 / macro-AUROC | 0.809 / 0.726 / 0.659 / 0.954 | 0.797 / 0.743 / 0.670 / 0.964 |
| Fitzpatrick17k: accuracy / balanced acc | 0.245 / 0.269 | 0.385 / 0.336 |
| Light (I-III) minus dark (IV-VI) balanced-acc gap, 95% bootstrap CI | 0.077 [0.023, 0.127] | 0.044 [-0.013, 0.104] |
| Max - min over skin types (ratio min/max) | 0.156 (0.55) | 0.121 (0.70) |

Gap CIs use 1,000 bootstrap resamples (`metrics.json`), as in section 4.1. `reports/figures/gap_comparison.png` was drawn
from the 2,000-resample comparison file, so its error bars differ slightly (e.g. ResNet-50: 0.022 to 0.131).

Balanced accuracy per Fitzpatrick type (n in brackets: I 450, II 672, III 380, IV 221, V 103, VI 34):

| FST | I | II | III | IV | V | VI |
|---|---|---|---|---|---|---|
| ResNet-50 | 0.350 | 0.276 | 0.193 | 0.208 | 0.202 | 0.203 |
| ViT-B/16 | 0.401 | 0.337 | 0.291 | 0.337 | 0.280 | 0.286 |

CIs for types V and VI are wide (e.g. ResNet VI: 0.113-0.389), see `reports/figures/fst_comparison.png`.
The drop is not one-directional: ResNet-50's type III (0.193) is already as low as types IV-VI, and ViT-B/16 scores the
same on type IV as on type II (0.337). The light-minus-dark gap therefore comes mostly from type I scoring higher, not from
a steady fall with darker skin.

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

### 4.1 Effect of oversampling (focal loss, seed 42)
Same test set (HAM10000 n = 1,441; Fitzpatrick17k n = 1,916) for all four runs. Gap = light-minus-dark balanced accuracy [95% CI].

| Run | Best val bal-acc | HAM test acc / bal-acc / macro-F1 | Melanoma recall | Fitzpatrick bal-acc | Gap [95% CI] |
|---|---|---|---|---|---|
| ResNet-50, no oversampling | 0.789 | 0.809 / 0.726 / 0.659 | 0.53 | 0.269 | 0.077 [0.023, 0.127] |
| ResNet-50 + oversampling | 0.794 | 0.750 / 0.717 / 0.651 | 0.60 | 0.260 | 0.022 [-0.039, 0.088] |
| ViT-B/16, no oversampling | 0.821 | 0.797 / 0.743 / 0.670 | 0.57 | 0.336 | 0.044 [-0.013, 0.104] |
| ViT-B/16 + oversampling | 0.820 | 0.809 / 0.758 / 0.711 | 0.64 | 0.327 | 0.060 [0.006, 0.116] |

Paired bootstrap on the gap (same 1,860 images with a known skin type): ResNet oversampling - baseline = -0.055, 95% CI [-0.128, 0.016], p = 0.13;
ViT oversampling - baseline = +0.016, CI [-0.041, 0.072], p = 0.55; ViT - ResNet (both with oversampling) = +0.038, CI [-0.041, 0.118], p = 0.37.
Figures for the oversampled pair: `reports/figures/oversampling/`. Explanation maps were generated only for the two baseline runs.

## 5. Discussion & limitations  *(Karanpal)*
What the numbers support:
- On HAM10000 the two models are close (balanced acc 0.726 vs 0.743); ViT has the higher validation score.
- On Fitzpatrick17k both models drop sharply (balanced acc 0.27 and 0.34, chance is about 0.14). This is mostly dermoscopic -> clinical **domain shift**, not only skin tone.
- The light-minus-dark gap point estimate is positive in all four runs (lower accuracy on dark skin), but its 95% CI excludes zero in only two of the four (ResNet-50 baseline and ViT-B/16 with oversampling), so the effect is not consistently distinguishable from noise at this sample size. The direct CNN-vs-ViT comparison on the baseline runs is **not significant** (p = 0.36), so we cannot claim that ViT is fairer; ViT is more accurate overall on Fitzpatrick17k.
- **Oversampling made no clear difference.** Validation balanced accuracy is unchanged (0.789 -> 0.794, 0.821 -> 0.820). On the HAM10000 test set balanced accuracy moved by -0.009 (ResNet) and +0.015 (ViT), and melanoma recall rose for both (0.53 -> 0.60, 0.57 -> 0.64), but ResNet's overall accuracy fell from 0.809 to 0.750. All gap differences caused by oversampling have CIs that include zero (p = 0.13 and 0.55). The gap CIs also flip sides of zero between runs (ResNet: above zero without, including zero with oversampling; ViT: the opposite), which shows that a single-seed gap is too noisy to rank the models on fairness.
- Behaviour worth noting (baseline runs): on Fitzpatrick17k ResNet-50 predicts `nv` for 41% of images (true share 12%), ViT-B/16 for 27%; on HAM10000 about 28% of melanomas (41 of 146) are predicted as `nv` by ResNet-50 and 19% (27) by ViT-B/16. Melanoma recall is only 0.53 and 0.57, which matters clinically. Oversampling moves the two models in opposite directions on Fitzpatrick17k: ResNet-50 predicts `nv` less often (26%), but ViT-B/16 more often (41%).
- **Explanation quality has no clear winner.** ViT-B/16 is better on deletion AUC (0.239 vs 0.310), while ResNet-50 is better on insertion AUC (0.544 vs 0.441). Both scores depend on each model's own probability scale, so this cross-model comparison is indicative only.
- Per-class recall by light vs dark (`per_class_recall_light_vs_dark` in `metrics.json`) shows no consistent pattern; `df` has only 8 dark-skin images and recall 0 for both models.

Limitations: the HAM10000 split depends on the scikit-learn version (Colab test n = 1,441 vs 1,431 locally), so numbers should be reproduced in a single environment; one seed and one training run per setting; default hyper-parameters (no search); only the focal-loss setting (no plain CE / weighted-CE); HAM10000 itself has no skin-type labels, so fairness is measured only on Fitzpatrick17k; the label mapping to 7 classes drops most conditions (1,916 of 16,577 images) and leaves squamous cell carcinoma and port-wine stain unmapped on purpose; skin types V and VI have only 103 and 34 images, and `df`/`bkl` are small. Explanation-quality numbers use 256 images (6 of type VI), and deletion/insertion AUC depends on each model's probability scale, so cross-model comparison is only indicative. Pointing-game / IoU were not computed (HAM10000 lesion masks exist but were not downloaded or wired in).

## 6. Conclusion
We built a reproducible pipeline (lesion-level splits, ResNet-50 and ViT-B/16 with a shared training loop,
Grad-CAM / attention rollout, per-skin-type evaluation with bootstrap CIs) and ran four focal-loss models
(ResNet-50 and ViT-B/16, each with and without oversampling). ViT-B/16 reaches a slightly higher balanced accuracy on the
HAM10000 test set (0.743 / 0.758) than ResNet-50 (0.726 / 0.717). On Fitzpatrick17k, which adds a dermoscopic-to-clinical
domain shift, all models drop strongly and the light-minus-dark gap is positive for all four (between 0.022 and 0.077), i.e. lower balanced accuracy on dark skin, but its 95% CI excludes zero in only two of the four runs.
None of the gap differences, neither between architectures nor caused by oversampling, is statistically significant, so
**our data do not show that either architecture is fairer, nor that oversampling helps fairness**; ViT-B/16 is clearly more
accurate on Fitzpatrick17k overall. Oversampling raised melanoma recall for both models but did not change validation
performance. Explanation-quality scores (deletion / insertion AUC, baseline runs only) are based on 256 images with very few
dark-skin samples, so they are indicative only. Future work: several seeds, a plain CE / weighted-CE comparison and a
hyper-parameter search with equal budget, a fixed saved split, lesion masks for pointing-game / IoU, and evaluation on a
dataset with dermoscopic images labelled by skin tone.

## References
1. Esteva et al., *Dermatologist-level classification of skin cancer with deep neural networks*, Nature 2017 - https://doi.org/10.1038/nature21056
2. Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR 2021 - https://arxiv.org/abs/2010.11929
3. Groh et al., *Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset*, CVPR-W 2021 - https://arxiv.org/abs/2104.09957
4. Tschandl et al., *The HAM10000 dataset*, Scientific Data 2018 - https://doi.org/10.1038/sdata.2018.161
5. Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017 (Grad-CAM)
6. Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017 (focal loss)
7. Abnar & Zuidema, *Quantifying Attention Flow in Transformers*, ACL 2020 (attention rollout)
8. Petsiuk et al., *RISE: Randomized Input Sampling for Explanation of Black-box Models*, BMVC 2018 (deletion / insertion)

# Fair & Explainable Skin-Lesion Classification: CNN vs Vision Transformer

> Draft. Numbers below come from one run per model (seed 42) logged in `results/EXPERIMENTS.md`
> (`results/*/*/metrics.json`, `reports/figures/fairness_comparison.json`). Introduction and conclusion are drafts for the
> team to edit; the oversampling / CE comparison was not run.

## 1. Introduction  *(draft, team to edit)*
Deep-learning classifiers for skin lesions are usually trained on dermoscopic data that is dominated by lighter
skin, so a model that looks accurate overall may work worse for patients with darker skin. We study the 7-class
HAM10000 lesion classification task with two model families, a CNN (ResNet-50) and a Vision Transformer
(ViT-B/16), and ask: **which one is fairer across Fitzpatrick skin types, and which gives better explanations
(Grad-CAM for the CNN, attention rollout for the ViT)?** HAM10000 has no skin-type labels, so fairness is measured on
Fitzpatrick17k, restricted to the conditions that map onto the 7 HAM classes. Both datasets are heavily imbalanced
(nv is 67% of HAM10000), which we address with focal loss. Related work: Esteva et al. (2017) for CNN
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
- Behaviour worth noting: on Fitzpatrick17k ResNet-50 predicts `nv` for 41% of images (true share 12%), ViT-B/16 for 27%; on HAM10000 about 28% of melanomas (41 of 146) are predicted as `nv` by ResNet-50 and 19% (27) by ViT-B/16. Melanoma recall is only 0.53 and 0.57, which matters clinically.
- Per-class recall by light vs dark (`per_class_recall_light_vs_dark` in `metrics.json`) shows no consistent pattern; `df` has only 8 dark-skin images and recall 0 for both models.

Limitations: the HAM10000 split depends on the scikit-learn version (Colab test n = 1,441 vs 1,431 locally), so numbers should be reproduced in a single environment; one seed and one training run per model; default hyper-parameters (no search); only the focal-loss setting; HAM10000 itself has no skin-type labels, so fairness is measured only on Fitzpatrick17k; the label mapping to 7 classes drops most conditions (1,916 of 16,577 images) and leaves squamous cell carcinoma and port-wine stain unmapped on purpose; skin types V and VI have only 103 and 34 images, and `df`/`bkl` are small. Explanation-quality numbers use 256 images (6 of type VI), and deletion/insertion AUC depends on each model's probability scale, so cross-model comparison is only indicative. Pointing-game / IoU were not computed (no lesion masks).

## 6. Conclusion  *(draft, team to edit)*
We built a reproducible pipeline (lesion-level splits, ResNet-50 and ViT-B/16 with a shared training loop,
Grad-CAM / attention rollout, per-skin-type evaluation with bootstrap CIs) and ran one focal-loss model of each type.
Both reach a similar balanced accuracy on the HAM10000 test set (0.726 ResNet-50, 0.743 ViT-B/16). On Fitzpatrick17k,
which adds a dermoscopic-to-clinical domain shift, both drop strongly and both perform worse on dark skin
(light-minus-dark gap 0.077 for ResNet-50, 0.044 for ViT-B/16). The difference between the two gaps is not
statistically significant, so **our data do not show that either architecture is fairer**; ViT-B/16 is
more accurate on Fitzpatrick17k overall. Explanation-quality scores (deletion / insertion AUC) are reported but are
based on 256 images with very few dark-skin samples, so they are indicative only. Future work: several seeds,
the CE / weighted-CE / oversampling comparison and a hyper-parameter search with equal budget, a fixed saved split,
lesion masks for pointing-game / IoU, and evaluation on a dataset with dermoscopic images labelled by skin tone.

## References
1. Esteva et al., *Dermatologist-level classification of skin cancer with deep neural networks*, Nature 2017 - https://doi.org/10.1038/nature21056
2. Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR 2021 - https://arxiv.org/abs/2010.11929
3. Groh et al., *Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset*, CVPR-W 2021 - https://arxiv.org/abs/2104.09957
4. Tschandl et al., *The HAM10000 dataset*, Scientific Data 2018 - https://doi.org/10.1038/sdata.2018.161
5. Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017 (focal loss)
6. Abnar & Zuidema, *Quantifying Attention Flow in Transformers*, ACL 2020 (attention rollout)
7. Petsiuk et al., *RISE: Randomized Input Sampling for Explanation of Black-box Models*, BMVC 2018 (deletion / insertion)

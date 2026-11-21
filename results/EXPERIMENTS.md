# Experiment log

| Date | Who | Model | Config / changes | Seed | Val bal-acc | Test bal-acc | FST gap | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-11-21 | Anamika (Colab T4) | ResNet-50 | configs/resnet50.yaml, focal (gamma 2, class-weighted alpha), no oversampling, lr 1e-4, bs 32, max 15 epochs, early-stop patience 5 (stopped at epoch 13, best epoch 8) | 42 | 0.789 | 0.726 | 0.077 [95% CI 0.023, 0.127] | Fitzpatrick17k (domain-shifted, n=1916) bal-acc 0.269; run `resnet50_focal_s42` |
| 2026-11-21 | Anamika (Colab T4) | ViT-B/16 | configs/vit_b16.yaml, focal (gamma 2, class-weighted alpha), no oversampling, lr 3e-5, layer decay 0.75, 3 warm-up epochs, bs 32, max 15 epochs, patience 5 (stopped at epoch 12, best epoch 7) | 42 | 0.821 | 0.743 | 0.044 [95% CI -0.013, 0.104] | Fitzpatrick17k bal-acc 0.336; run `vit_b16_focal_s42` |

FST gap = light (I-III) minus dark (IV-VI) balanced accuracy on Fitzpatrick17k; positive = worse on dark skin.
Hyper-parameters are the config defaults (no HP search run). Only focal loss, 1 seed; CE / weighted-CE / oversampling variants not run.

# This package is the "explain" step of the pipeline (data -> model -> train -> evaluate -> explain).
# A classifier only gives a label. Explainability tools show WHY, by drawing heat-maps
# (bright spots mark the image areas that mattered most to the model's decision).
# It runs after training, on a saved checkpoint; scripts/explain.py is the only caller.
# Submodules:
#   gradcam: heat-maps showing which image parts a CNN used.
#   attention: attention rollout heat-maps for a Vision Transformer (ViT).
#   quality: scores that check how good those heat-maps are.
# This file has no code; it just marks the folder as a Python package.
# The package needs no code, so this single "pass" statement does nothing.
pass

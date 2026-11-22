# This file is the root of the skinfair package.
# It only stores the list of diagnosis classes that every other file imports.
# A class's position in the list is its number (label), so the order must never change.

# The 7 diagnosis codes of the HAM10000 dataset; the list position is the integer label.
# akiec = actinic keratosis, bcc = basal cell carcinoma, bkl = benign keratosis,
# df = dermatofibroma, mel = melanoma, nv = mole (nevus), vasc = vascular lesion.
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
# How many classes there are; the model needs one output score per class.
NUM_CLASSES = len(CLASSES)

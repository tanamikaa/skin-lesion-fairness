"""skinfair — Fair & explainable skin-lesion classification (P01).

The package root only defines the list of diagnosis classes shared by every module.
"""

# The 7 HAM10000 diagnosis codes. The list index is the integer label used everywhere:
# akiec = actinic keratosis / Bowen's, bcc = basal cell carcinoma, bkl = benign keratosis,
# df = dermatofibroma, mel = melanoma, nv = melanocytic nevus, vasc = vascular lesion.
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
# Number of output units of the classifier head (7).
NUM_CLASSES = len(CLASSES)

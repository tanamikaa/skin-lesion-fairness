# This file translates Fitzpatrick17k condition names into the 7 HAM10000 class codes.
# The model is trained on HAM10000 but also tested on Fitzpatrick17k, which uses its own
# 114 condition names. To score the model there, each name must become one of our 7 codes.
# Any label not listed in the table below is dropped (it has no HAM match).
# Notes: squamous cell carcinoma is NOT mapped to akiec, and port wine stain is NOT mapped
# to vasc, because they are different diseases.

# lets us write the `str | None` type hint on older Python versions
from __future__ import annotations

# Counter counts how many times each value appears
from collections import Counter

# lookup table: Fitzpatrick17k label (lower case) -> HAM10000 class code.
# Keys must be lower case because normalise_label lower-cases labels before the lookup.
FITZ_TO_HAM = {
    # melanoma names all map to "mel"
    "melanoma": "mel",
    "malignant melanoma": "mel",
    "superficial spreading melanoma ssm": "mel",
    "lentigo maligna": "mel",
    # basal cell carcinoma names map to "bcc"
    "basal cell carcinoma": "bcc",
    "basal cell carcinoma morpheiform": "bcc",
    "solid cystic basal cell carcinoma": "bcc",
    # actinic keratosis maps to "akiec"
    "actinic keratosis": "akiec",
    # seborrheic keratosis maps to "bkl" (benign keratosis-like lesions)
    "seborrheic keratosis": "bkl",
    # dermatofibroma maps to "df"
    "dermatofibroma": "df",
    # nevus (mole) names map to "nv"
    "nevocytic nevus": "nv",
    "congenital nevus": "nv",
    "halo nevus": "nv",
    # pyogenic granuloma names map to "vasc" (vascular lesions)
    "pyogenic granuloma": "vasc",
    "granuloma pyogenic": "vasc",
}


# Cleans a label so it can be looked up: text, no side spaces, lower case.
# Args: label: a label (usually a string, but NaN or a number also works).
# Returns: a clean string, e.g. "  Melanoma " becomes "melanoma".
# Why: small differences in case or spaces would otherwise cause missed matches.
def normalise_label(label) -> str:
    """Cleans a label so it can be looked up in the mapping table.
    It turns the label into text, so NaN and numbers do not break it.
    It removes spaces at the start and end.
    It changes all letters to lower case.
    Input: label, usually a string.
    Output: a clean string. It is needed so small differences do not cause missed matches.
    """
    # str() makes NaN and numbers safe, strip removes side spaces, lower makes lower case
    return str(label).strip().lower()


# Finds the HAM class for a Fitzpatrick17k label.
# Args: label: a Fitzpatrick17k condition name.
# Returns: one of the 7 class codes (like "mel"), or None if there is no match.
# Example: map_label("Halo Nevus") gives "nv"; map_label("psoriasis") gives None.
def map_label(label) -> str | None:
    """Finds the HAM10000 class code for a Fitzpatrick17k condition name.
    Step 1: clean the label with normalise_label.
    Step 2: look the clean text up in the FITZ_TO_HAM table.
    Step 3: if it is not in the table, give back None.
    Input: label, a condition name.
    Output: a class code like "mel", or None. It is needed to score the model on Fitzpatrick17k.
    """
    # clean the label, then look it up; .get gives None when it is not in the table
    return FITZ_TO_HAM.get(normalise_label(label))


# Summarises how a Fitzpatrick17k table maps to HAM classes.
# Args: df: the Fitzpatrick17k table.
#       label_col: name of the column that holds the condition names.
# Returns: a dict with kept and dropped row counts, rows per HAM class, mapping keys not
#          found in the file (possible typos), and the dropped labels with counts.
# Why: a quick check that the mapping table matches the real data.
def mapping_report(df, label_col: str = "label") -> dict:
    """Makes a summary of how a Fitzpatrick17k table maps to HAM classes.
    Step 1: clean every label and map it to a HAM class.
    Step 2: count rows kept (mapped) and rows dropped (no match).
    Step 3: count rows per class and list mapping keys never seen in the file.
    Input: df, the table, and label_col, the column with condition names.
    Output: a report dict. It is needed to check the mapping fits the real data.
    """
    # clean label for every row
    norm = df[label_col].map(normalise_label)
    # HAM class for every row (NaN where there is no match)
    mapped = norm.map(FITZ_TO_HAM)
    # the set of all different labels that exist in the file
    seen = set(norm.unique())
    # build and return the report dict
    return {
        # number of rows that have a HAM class
        "kept": int(mapped.notna().sum()),
        # number of rows that have no HAM class
        "dropped": int(mapped.isna().sum()),
        # number of rows for each HAM class
        "per_class": Counter(mapped.dropna()),
        # mapping keys that never appear in the file (maybe typos)
        "missing_keys": sorted(k for k in FITZ_TO_HAM if k not in seen),
        # dropped labels with how many times each appears
        "unmapped": Counter(norm[mapped.isna()]),
    }

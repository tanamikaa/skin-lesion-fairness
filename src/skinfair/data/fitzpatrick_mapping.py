"""Map Fitzpatrick17k `label` strings to the 7 HAM10000 classes. Owner: Afzal Raza.

Anything not listed is dropped (Fitzpatrick17k has 114 conditions, most have no HAM equivalent).

Decisions worth knowing (report them honestly):
* `squamous cell carcinoma` is NOT mapped to `akiec`: HAM's akiec is actinic keratosis /
  Bowen's disease (in-situ), whereas invasive SCC is a different entity.
* `port wine stain` is NOT mapped to `vasc`: it is a capillary malformation, not an angioma.
* Keys were verified against the real fitzpatrick17k.csv (16,577 rows): 1,916 rows map.
  Re-check with `mapping_report` if the CSV version changes.
"""
from __future__ import annotations

from collections import Counter

FITZ_TO_HAM = {
    # melanoma
    "melanoma": "mel",
    "malignant melanoma": "mel",
    "superficial spreading melanoma ssm": "mel",
    "lentigo maligna": "mel",
    # basal cell carcinoma
    "basal cell carcinoma": "bcc",
    "basal cell carcinoma morpheiform": "bcc",
    "solid cystic basal cell carcinoma": "bcc",
    # actinic keratoses / intraepithelial carcinoma
    "actinic keratosis": "akiec",
    # benign keratosis-like lesions
    "seborrheic keratosis": "bkl",
    # dermatofibroma
    "dermatofibroma": "df",
    # melanocytic nevi
    "nevocytic nevus": "nv",
    "congenital nevus": "nv",
    "halo nevus": "nv",
    # vascular lesions
    "pyogenic granuloma": "vasc",
    "granuloma pyogenic": "vasc",
}


def normalise_label(label) -> str:
    """Turn a raw label into a lookup key: text, no outer spaces, lower case.

    Input: any value (str() handles NaN or numbers). Returns the cleaned string.
    """
    return str(label).strip().lower()


def map_label(label) -> str | None:
    """Return the HAM class for a Fitzpatrick17k label, or None if it is unmapped."""
    return FITZ_TO_HAM.get(normalise_label(label))  # .get returns None for unknown keys


def mapping_report(df, label_col: str = "label") -> dict:
    """Summarise how a fitzpatrick17k dataframe maps onto HAM classes.

    Returns dict with: kept (rows kept), dropped (rows dropped), per_class (Counter of
    HAM classes), missing_keys (mapping keys never seen in the CSV - likely typos),
    unmapped (Counter of labels dropped).
    Input: DataFrame with a label column (name given by label_col).
    """
    norm = df[label_col].map(normalise_label)  # cleaned label per row
    mapped = norm.map(FITZ_TO_HAM)  # HAM class per row, NaN where not in the mapping
    seen = set(norm.unique())  # all distinct cleaned labels in the CSV
    return {
        "kept": int(mapped.notna().sum()),
        "dropped": int(mapped.isna().sum()),
        "per_class": Counter(mapped.dropna()),
        "missing_keys": sorted(k for k in FITZ_TO_HAM if k not in seen),
        "unmapped": Counter(norm[mapped.isna()]),
    }

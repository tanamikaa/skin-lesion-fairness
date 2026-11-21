"""Map Fitzpatrick17k `label` strings to the 7 HAM10000 classes. Owner: Afzal Raza.

Keys are matched case-insensitively against the `label` column of
fitzpatrick17k.csv (columns: md5hash, fitzpatrick_scale,
fitzpatrick_centaur, label, nine_partition_label, three_partition_label,
qc, url, url_alphanum). Entries below were confirmed against the first
~1% of the actual CSV (https://github.com/mattgroh/fitzpatrick17k);
anything not listed is dropped.

TODO(Afzal): once the full CSV is downloaded locally, run
`python scripts/inspect_fitzpatrick_labels.py --csv data/fitzpatrick17k/fitzpatrick17k.csv`
to list every unique label and extend the "nv" / "vasc" entries below --
the ones here are a confirmed-but-partial start, not the full set.
"""
FITZ_TO_HAM = {
    "melanoma": "mel",
    "basal cell carcinoma": "bcc",
    "actinic keratosis": "akiec",
    "seborrheic keratosis": "bkl",
    "dermatofibroma": "df",
    # nv (melanocytic nevus) -- confirmed label strings; list is partial,
    # see TODO above for how to find the rest.
    "nevocytic nevus": "nv",
    "congenital nevus": "nv",
    "epidermal nevus": "nv",
    "naevus comedonicus": "nv",
    # vasc (vascular lesions) -- confirmed label strings; list is partial.
    "granuloma pyogenic": "vasc",
}

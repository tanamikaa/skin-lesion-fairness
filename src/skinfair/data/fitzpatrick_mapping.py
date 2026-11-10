"""Map Fitzpatrick17k `label` strings to the 7 HAM10000 classes. Owner: Afzal Raza.

TODO(Afzal): complete after inspecting fitzpatrick17k.csv; anything not listed is dropped.
"""
FITZ_TO_HAM = {
    "melanoma": "mel",
    "basal cell carcinoma": "bcc",
    "actinic keratosis": "akiec",
    "seborrheic keratosis": "bkl",
    "dermatofibroma": "df",
    # TODO: entries for "nv" and "vasc"
}

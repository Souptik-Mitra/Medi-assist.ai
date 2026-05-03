"""
Map free-text symptom descriptions + age/gender into the feature space of
synthetic_medical_symptoms_dataset.csv for the trained classifier.
"""
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

# Ordinal symptom columns (0–3 in training data)
ORDINAL_FEATURES: List[str] = [
    "fever",
    "cough",
    "fatigue",
    "headache",
    "muscle_pain",
    "nausea",
    "vomiting",
    "diarrhea",
    "skin_rash",
    "loss_smell",
    "loss_taste",
]

# Lab / vitals (float); default to training medians when not parsed from text
CONTINUOUS_FEATURES: List[str] = [
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "temperature_c",
    "oxygen_saturation",
    "wbc_count",
    "hemoglobin",
    "platelet_count",
    "crp_level",
    "glucose_level",
]

# (feature_name, list of substrings to search in lowercased text)
KEYWORD_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("fever", ("fever", "temperature", "chills", "febrile")),
    ("cough", ("cough", "wheez", "phlegm")),
    ("fatigue", ("fatigue", "tired", "exhaust", "weakness", "letharg")),
    ("headache", ("headache", "migraine")),
    ("muscle_pain", ("muscle", "body ache", "myalgia", "joint pain")),
    ("nausea", ("nausea", "queasy")),
    ("vomiting", ("vomit", "throwing up", "emesis")),
    ("diarrhea", ("diarrhea", "loose stool")),
    ("skin_rash", ("rash", "hives")),
    ("loss_smell", ("loss of smell", "anosmia", "can't smell", "cannot smell")),
    ("loss_taste", ("loss of taste", "ageusia", "can't taste", "cannot taste")),
]


def _severity_near_keyword(text: str, keyword: str) -> int:
    idx = text.find(keyword)
    if idx < 0:
        return 2
    chunk = text[max(0, idx - 55) : idx + len(keyword) + 55]
    if re.search(
        r"\b(severe|high|very bad|intense|104|105|106|39\.[5-9]|40\.|41\.)\b", chunk
    ):
        return 3
    if re.search(r"\b(mild|slight|low grade|low-grade)\b", chunk):
        return 1
    return 2


def _parse_float_after(text: str, patterns: Tuple[str, ...]) -> Optional[float]:
    for p in patterns:
        m = re.search(rf"{re.escape(p)}\s*(?:of|:)?\s*([0-9]+(?:\.[0-9]+)?)", text)
        if m:
            return float(m.group(1))
    return None


def features_from_text(
    symptoms_text: str,
    age: Any,
    gender: Optional[str],
    medians: Mapping[str, float],
    mode_gender: str = "Female",
) -> Dict[str, Any]:
    """Build one row of model features from form + text."""
    t = (symptoms_text or "").lower()
    row: Dict[str, Any] = {}

    try:
        row["age"] = int(float(str(age).strip())) if age not in (None, "", "Not provided") else int(round(medians["age"]))
    except (TypeError, ValueError, KeyError):
        row["age"] = 40

    g = (gender or "").strip()
    if g not in ("Male", "Female"):
        g = mode_gender if mode_gender in ("Male", "Female") else "Female"
    row["gender"] = g

    for col in ORDINAL_FEATURES:
        level = 0
        matched = False
        for feat, keys in KEYWORD_RULES:
            if feat != col:
                continue
            for k in keys:
                if k in t:
                    level = _severity_near_keyword(t, k)
                    matched = True
                    break
            if matched:
                break
        row[col] = level

    for col in CONTINUOUS_FEATURES:
        row[col] = float(medians[col])

    temp = _parse_float_after(t, ("temperature", "temp"))
    if temp is not None:
        if temp > 45:
            temp = (temp - 32) * 5 / 9
        row["temperature_c"] = temp

    spo2 = _parse_float_after(t, ("oxygen", "spo2", "o2 sat", "saturation"))
    if spo2 is not None:
        row["oxygen_saturation"] = min(100.0, max(70.0, spo2))

    hr = _parse_float_after(t, ("heart rate", "pulse", "bpm"))
    if hr is not None:
        row["heart_rate"] = hr

    sys_bp = _parse_float_after(t, ("systolic", "bp", "blood pressure"))
    if sys_bp is not None and sys_bp < 250:
        row["systolic_bp"] = sys_bp

    return row

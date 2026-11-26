import numpy as np
import pandas as pd

STRIDE_LABELS = [
    "Spoofing",
    "Tampering",
    "Repudiation",
    "Info_Disclosure",
    "DoS",
    "EoP",
]

STRIDE_CODES = ["S", "T", "R", "I", "D", "E"]

ATTACK_TO_STRIDE = {
    "normal": [0, 0, 0, 0, 0, 0],
    "generic": [0, 0, 0, 0, 1, 0],
    "exploits": [0, 1, 0, 0, 0, 1],
    "fuzzers": [0, 1, 0, 1, 0, 0],
    "dos": [0, 0, 0, 0, 1, 0],
    "reconnaissance": [1, 0, 0, 1, 0, 0],
    "analysis": [0, 0, 0, 1, 0, 0],
    "backdoor": [1, 0, 1, 0, 0, 1],
    "backdoors": [1, 0, 1, 0, 0, 1],
    "shellcode": [0, 1, 0, 0, 0, 1],
    "worms": [0, 1, 0, 0, 1, 0],
}


def normalize_attack_cat(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("normal").astype(str).str.strip().str.lower()
    return cleaned.replace({"backdoors": "backdoor"})


def encode_stride_targets(attack_cat: pd.Series) -> np.ndarray:
    normalized = normalize_attack_cat(attack_cat)
    unknown = set(normalized.unique()) - set(ATTACK_TO_STRIDE.keys())
    if unknown:
        raise ValueError(f"Unmapped attack categories encountered: {unknown}")
    vectors = [ATTACK_TO_STRIDE[cat] for cat in normalized]
    return np.asarray(vectors, dtype=np.float32)
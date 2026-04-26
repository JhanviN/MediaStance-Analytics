"""
CAMEO event code descriptions and groupings.
Source: CAMEO Manual 1.1b3 — https://parusanalytics.com/eventdata/cameo.dir/
"""

from __future__ import annotations

# Full CAMEO root code → human-readable description
CAMEO_DESCRIPTIONS: dict[str, str] = {
    "01": "Make public statement",
    "02": "Appeal",
    "03": "Express intent to cooperate",
    "04": "Consult",
    "05": "Engage in diplomatic cooperation",
    "06": "Engage in material cooperation",
    "07": "Provide aid",
    "08": "Yield",
    "09": "Investigate",
    "10": "Demand",
    "11": "Disapprove",
    "12": "Reject",
    "13": "Threaten",
    "14": "Protest",
    "15": "Exhibit force posture",
    "16": "Reduce relations",
    "17": "Coerce",
    "18": "Assault",
    "19": "Fight",
    "20": "Engage in mass violence",
}

# Map raw code (int or string) → normalized 2-digit string
def normalize_code(code: str | int) -> str:
    return str(int(str(code).strip())).zfill(2)


def get_description(code: str | int) -> str:
    return CAMEO_DESCRIPTIONS.get(normalize_code(code), f"Event-{code}")


def get_label(code: str | int) -> str:
    try:
        root = int(str(code).strip())
    except ValueError:
        return "neutral"
    if root <= 8:
        return "cooperative"
    if root <= 11:
        return "neutral"
    return "adversarial"


# Groupings for causality graph display
CAMEO_GROUPS: dict[str, list[str]] = {
    "Diplomatic Cooperation": ["03", "04", "05"],
    "Material Cooperation":   ["06", "07", "08"],
    "Public Statement":       ["01", "02"],
    "Investigation/Demand":   ["09", "10"],
    "Disapproval/Rejection":  ["11", "12"],
    "Threat/Coercion":        ["13", "15", "17"],
    "Protest/Reduce":         ["14", "16"],
    "Violence":               ["18", "19", "20"],
}

# Reverse lookup: code → group name
CODE_TO_GROUP: dict[str, str] = {
    code: group
    for group, codes in CAMEO_GROUPS.items()
    for code in codes
}

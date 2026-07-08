# config.py

SUBREDDITS = [
    "cancer",
    "melanoma",
    "lungcancer",
    "breastcancer",
    "coloncancer",
    "lymphoma",
    "leukemia",
    "CaregiverSupport",
]

# Broad immunotherapy search terms.
# These are used to collect posts from Arctic Shift.
SEARCH_TERMS = [
    "keytruda",
    "pembrolizumab",
    "opdivo",
    "nivolumab",
    "yervoy",
    "ipilimumab",
    "tecentriq",
    "atezolizumab",
    "imfinzi",
    "durvalumab",
    "immunotherapy",
    "immune therapy",
    "checkpoint inhibitor",
    "immune checkpoint inhibitor",
    "pd-1",
    "pd1",
    "pd-l1",
    "pdl1",
    "car-t",
    "car t",
    "CAR T",
]

DATE_WINDOWS = [
    ("2025-01-01", "2026-01-01"),
    ("2024-01-01", "2025-01-01"),
    ("2023-01-01", "2024-01-01"),
]

POST_LIMIT_PER_SEARCH = 25
REQUEST_SLEEP_SECONDS = 5


# Ollama local API endpoint.
OLLAMA_URL = "http://localhost:11434/api/generate"
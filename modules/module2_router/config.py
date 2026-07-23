"""
Configuration file for Module 2 - Region Router
"""

from pathlib import Path


# ==========================
# Project Directories
# ==========================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "outputs"
DEBUG_DIR = OUTPUT_DIR / "debug"

SEGMENTATION_JSON = INPUT_DIR / "segmentation_output.json"

OUTPUT_JSON = OUTPUT_DIR / "routed_regions.json"


# ==========================
# Confidence Thresholds
# ==========================

MIN_CONFIDENCE = 0.50

# Below this → discard
DISCARD_THRESHOLD = 0.30

# Above this → accept directly
HIGH_CONFIDENCE = 0.90


# ==========================
# Reading Order
# ==========================

SORT_TOP_TO_BOTTOM = True
SORT_LEFT_TO_RIGHT = True


# ==========================
# Region Routing Map
# ==========================

ROUTING_TABLE = {
    "paragraph": "ocr",
    "text": "ocr",
    "heading": "ocr",
    "question": "ocr",

    "equation": "math_ocr",

    "diagram": "diagram_parser",
    "flowchart": "diagram_parser",
    "block_diagram": "diagram_parser",

    "table": "table_parser",

    "graph": "graph_parser",

    "crossed_out": "ignore",

    "unknown": "manual_review"
}


# ==========================
# Supported Labels
# ==========================

VALID_LABELS = set(ROUTING_TABLE.keys())


# ==========================
# Output Settings
# ==========================

SAVE_DEBUG_JSON = True
VERBOSE = True


# ==========================
# Create folders automatically
# ==========================

OUTPUT_DIR.mkdir(exist_ok=True)
DEBUG_DIR.mkdir(exist_ok=True)
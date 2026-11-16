"""Shared constants for the Insurance Claim Cycle-Time & Documentation Monitor."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
TRUTH_DIR = ROOT / "data" / "truth"      # only used to self-test the reconstruction
DB_PATH = ROOT / "claims.db"

# Snapshot date: every open claim is aged against this moment.
AS_OF = pd.Timestamp("2026-11-12 00:00")

# Canonical workflow stages, in order. "Closed" is terminal (no duration).
STAGES = [
    "Reported",
    "Registered",
    "Document Collection",
    "Assessment",
    "Settlement Processing",
    "Payment Issued",
    "Closed",
]
STAGE_ORDER = {s: i for i, s in enumerate(STAGES)}
WORK_STAGES = STAGES[:-1]  # stages that have a duration

# Default service targets (days a claim should spend in each stage). Editable in the app.
DEFAULT_SLA_DAYS = {
    "Reported": 1,
    "Registered": 2,
    "Document Collection": 10,
    "Assessment": 10,
    "Settlement Processing": 5,
    "Payment Issued": 3,
}

# A claim with no activity for this many days is flagged as stalled.
STALLED_DAYS = 21

# Chart palette (validated reference palette, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

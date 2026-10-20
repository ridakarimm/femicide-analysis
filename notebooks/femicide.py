"""Loading and cleaning helpers shared by the analysis notebooks."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("FEMICIDE_DATA", REPO / "data" / "raw"))
FIGURES = REPO / "figures"

# Identifying and free-text fields, dropped on load so they never reach an output cell.
PII_COLUMNS = [
    "HER NAME", "HER PHOTO", "ABOUT HER", "SUSPECT / POI NAME", "HIS PHOTO", "HIS OCCUPATION",
    "DETAILS / LENGTH", "DEATH DETAILS", "DETAILS & POST-MORTEM INJURIES / ACTIONS",
    "ADDITIONAL VICTIMS IN SAME INCIDENT", "SEXUALLY MOTIVATED / RELATED ACTIONS & HISTORY",
    "CHILD-RELATED / CUSTODY ISSUES", "CASE STATUS / DETAILS", "SENTENCE", "YEARS / DETAILS",
    "DOCUMENTS", "SOURCES", "CASE KEYWORDS", "KEYWORD DETAILS", "DETAILS / NOTES", "SYNOPSIS",
    "ADDRESS FOR MAPPING", "LATITUDE / LONGITUDE", "ATTACHMENTS SAVED", "DISCLAIMER",
    "REQUESTS FOR HIDING / CORRECTING / INCLUDING INFO",
]


def clean_text(s):
    """Strip whitespace and stray quotes, and treat blanks as missing."""
    s = s.astype("string").str.replace('"', "", regex=False).str.strip()
    return s.mask(s == "")


def load_cases(data_dir=DATA_DIR):
    """All yearly Women Count USA exports (2015.csv ... 2024.csv) as one de-identified table."""
    files = sorted(Path(data_dir).glob("20[0-9][0-9].csv"))
    if not files:
        raise FileNotFoundError(f"No yearly CSVs in {data_dir}. See the README for how to get the data.")
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    df = df.drop(columns=[c for c in df.columns if c.strip() in PII_COLUMNS or "COLLECTIVE VIOLENCE" in c])

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce", utc=True).dt.tz_localize(None)
    # Cases known only by year are recorded on Dec 31, so they carry no month information.
    df["DATE_IS_PLACEHOLDER"] = df["DATE"].dt.strftime("%m-%d").eq("12-31")
    for col in df.columns.drop(["DATE", "DATE_IS_PLACEHOLDER"]):
        if df[col].dtype == object:
            df[col] = clean_text(df[col])
    df["STATE"] = df["STATE"].str.upper()
    df["HER AGE"] = pd.to_numeric(df["HER AGE"], errors="coerce")
    df["HIS AGE"] = pd.to_numeric(df["HIS AGE"], errors="coerce")
    return df


def load_population():
    """2020 Census resident population by two-letter state code."""
    pop = pd.read_csv(REPO / "data" / "state_population.csv")
    return pop.set_index("state_code")["2020_census"]


def top_categories(s, k, other="Other", missing="Not recorded"):
    """Title-case a categorical column and keep its k most common values."""
    s = s.str.title().fillna(missing)
    keep = s.value_counts().index[:k]
    return s.where(s.isin(keep), other)


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=200, bbox_inches="tight")

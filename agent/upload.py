"""Lecture tolérante des CSV d'export PME françaises (latin-1, séparateur ';')."""
from __future__ import annotations

import csv
import io

import pandas as pd


def _decode(raw: bytes) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Fichier illisible : encodage non reconnu (ni utf-8 ni latin-1).")


def read_uploaded_csv(raw_bytes: bytes) -> pd.DataFrame:
    text = _decode(raw_bytes)
    head = "\n".join(text.splitlines()[:5])
    try:
        sep = csv.Sniffer().sniff(head, delimiters=";,").delimiter
    except csv.Error:
        sep = ";"
    df = pd.read_csv(io.StringIO(text), sep=sep)
    if df.shape[1] == 1 and sep != ";":
        df = pd.read_csv(io.StringIO(text), sep=";")
    return df


def validate_dataframe(df: pd.DataFrame) -> None:
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError("Fichier invalide : aucune donnée exploitable (0 ligne ou 0 colonne).")

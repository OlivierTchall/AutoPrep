"""fix_column_types — outil de modification. Parsing tolérant multi-format pour les dates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tools.registry import envelope

_DATE_PATTERNS = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%m/%d/%Y"]


def parse_dates_tolerant(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    remaining = s.notna()
    for fmt in _DATE_PATTERNS:
        if not remaining.any():
            break
        parsed = pd.to_datetime(s[remaining], format=fmt, errors="coerce")
        got = parsed.notna()
        out.loc[parsed.index[got]] = parsed[got]
        remaining.loc[parsed.index[got]] = False
    if remaining.any():
        parsed = pd.to_datetime(s[remaining], errors="coerce", dayfirst=True)
        out.loc[parsed.index] = parsed
    return out


def _coerce(series: pd.Series, target: str) -> pd.Series:
    if target == "date":
        return parse_dates_tolerant(series)
    if target == "entier":
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    if target == "decimal":
        return pd.to_numeric(series, errors="coerce").astype("float64")
    if target == "texte":
        return series.astype("string")
    if target == "categorie":
        return series.astype("category")
    raise ValueError(f"type cible inconnu: {target}")


def fix_column_types(df, colonnes, types_cibles, justification):
    new = df.copy()
    avant, apres, non_convertis = {}, {}, {}
    for col in colonnes:
        if col not in new.columns:
            return df, envelope("error", f"Colonne inexistante : {col}.", detail={"colonne": col})
        target = types_cibles.get(col)
        if target is None:
            return df, envelope("error", f"Aucun type cible fourni pour {col}.")
        avant[col] = str(df[col].dtype)
        before_na = int(new[col].isna().sum())
        new[col] = _coerce(new[col], target)
        after_na = int(new[col].isna().sum())
        apres[col] = str(new[col].dtype)
        non_convertis[col] = max(0, after_na - before_na)
    total_bad = sum(non_convertis.values())
    return new, envelope(
        "ok",
        f"Types corrigés pour {', '.join(colonnes)}. {total_bad} valeur(s) non convertible(s) → manquantes.",
        metrics={"avant": avant, "apres": apres, "non_convertis": non_convertis},
    )

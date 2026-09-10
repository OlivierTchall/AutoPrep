"""detect_outliers (lecture) + treat_outliers (modification)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

import config
from tools.registry import envelope


def _fences(s: pd.Series, methode: str):
    s = s.dropna()
    if methode == "zscore":
        m, sd = s.mean(), s.std()
        return m - config.OUTLIER_Z * sd, m + config.OUTLIER_Z * sd
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - config.OUTLIER_IQR_K * iqr, q3 + config.OUTLIER_IQR_K * iqr


def detect_outliers(df, colonnes, methode="iqr"):
    detail = {}
    for c in colonnes:
        if c not in df.columns or not is_numeric_dtype(df[c]):
            return envelope("error", f"Colonne absente ou non numérique : {c}.")
        low, high = _fences(df[c], methode)
        mask = (df[c] < low) | (df[c] > high)
        detail[c] = {
            "n_outliers": int(mask.sum()),
            "bornes": [round(float(low), 4), round(float(high), 4)],
            "exemples": df.loc[mask, c].head(5).tolist(),
        }
    total = sum(v["n_outliers"] for v in detail.values())
    return envelope("ok", f"{total} valeur(s) extrême(s) détectée(s) ({methode}).", metrics={"total": total}, detail=detail)


def _target_mask(s: pd.Series, methode: str, bornes: dict | None):
    if bornes:
        lo = bornes.get("min")
        hi = bornes.get("max")
        m = pd.Series(False, index=s.index)
        if lo is not None:
            m |= s < lo
        if hi is not None:
            m |= s > hi
        return m, lo, hi
    low, high = _fences(s, methode)
    return (s < low) | (s > high), low, high


def treat_outliers(df, colonnes, action, justification, methode="iqr", bornes_valides=None):
    for c in colonnes:
        if c not in df.columns or not is_numeric_dtype(df[c]):
            return df, envelope("error", f"Colonne absente ou non numérique : {c}.")
    new = df.copy()
    lignes_avant = len(new)
    avant = {c: {"min": float(df[c].min()), "max": float(df[c].max())} for c in colonnes}
    cibles = {}
    drop_mask = pd.Series(False, index=new.index)
    for c in colonnes:
        mask, lo, hi = _target_mask(new[c], methode, bornes_valides)
        cibles[c] = int(mask.sum())
        if action == "cap":
            if lo is not None:
                new.loc[mask & (new[c] < lo), c] = lo
            if hi is not None:
                new.loc[mask & (new[c] > hi), c] = hi
        elif action == "set_nan":
            new.loc[mask, c] = np.nan
        elif action == "remove_rows":
            drop_mask |= mask
    if action == "remove_rows":
        new = new.loc[~drop_mask].reset_index(drop=True)
    apres = {c: {"min": float(new[c].min()), "max": float(new[c].max())} for c in colonnes}
    return new, envelope(
        "ok",
        f"Valeurs extrêmes traitées ({action}) sur {', '.join(colonnes)}.",
        metrics={"avant": avant, "apres": apres, "lignes_avant": lignes_avant,
                 "lignes_apres": len(new), "cibles": cibles},
    )

"""profile_dataset — lecture seule. Types, manquants, doublons, stats, min/max par colonne numérique."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tools.registry import envelope


def profile_dataset(df: pd.DataFrame) -> dict:
    n = len(df)
    manquants = {
        c: {"n": int(df[c].isna().sum()), "taux": round(float(df[c].isna().mean()), 4)}
        for c in df.columns
    }
    num_cols = df.select_dtypes(include=[np.number]).columns
    numeriques = {
        c: {
            "min": float(df[c].min()),
            "max": float(df[c].max()),
            "mean": round(float(df[c].mean()), 4),
            "std": round(float(df[c].std()), 4),
            "median": float(df[c].median()),
        }
        for c in num_cols
    }
    n_dup = int(df.duplicated().sum())
    negs = [c for c, s in numeriques.items() if s["min"] < 0]
    phrase = f"{n} lignes, {len(df.columns)} colonnes, {n_dup} doublon(s)."
    if negs:
        phrase += f" Valeurs négatives suspectes dans : {', '.join(negs)} (règle métier)."
    return envelope(
        "ok",
        phrase,
        metrics={"n_lignes": n, "n_doublons": n_dup},
        detail={
            "n_lignes": n,
            "n_colonnes": len(df.columns),
            "types": {c: str(t) for c, t in df.dtypes.items()},
            "valeurs_manquantes": manquants,
            "n_doublons": n_dup,
            "numeriques": numeriques,
            "apercu": df.head(5).to_dict(orient="records"),
        },
    )

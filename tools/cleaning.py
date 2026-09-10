"""handle_duplicates + handle_missing_values — outils de modification."""
from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype

from tools.registry import envelope


def handle_duplicates(df, sous_ensemble_colonnes, justification):
    subset = sous_ensemble_colonnes or None
    if subset:
        missing = [c for c in subset if c not in df.columns]
        if missing:
            return df, envelope("error", f"Colonnes inexistantes : {missing}.")
    before = len(df)
    new = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    after = len(new)
    return new, envelope(
        "ok",
        f"{before - after} doublon(s) supprimé(s) ({before} → {after} lignes).",
        metrics={"avant": before, "apres": after, "supprimees": before - after},
    )


def handle_missing_values(df, colonnes, strategie, valeur_constante, justification):
    missing = [c for c in colonnes if c not in df.columns]
    if missing:
        return df, envelope("error", f"Colonnes inexistantes : {missing}.")
    avant = {c: int(df[c].isna().sum()) for c in colonnes}
    new = df.copy()
    lignes_avant = len(new)

    if strategie == "drop_rows":
        new = new.dropna(subset=colonnes).reset_index(drop=True)
    elif strategie in ("mean", "median"):
        for c in colonnes:
            if not is_numeric_dtype(new[c]):
                return df, envelope("error", f"Stratégie '{strategie}' impossible sur la colonne non numérique {c}.")
            val = new[c].mean() if strategie == "mean" else new[c].median()
            new[c] = new[c].fillna(val)
    elif strategie == "mode":
        for c in colonnes:
            m = new[c].mode(dropna=True)
            if not m.empty:
                new[c] = new[c].fillna(m.iloc[0])
    elif strategie == "constant":
        for c in colonnes:
            new[c] = new[c].fillna(valeur_constante)

    apres = {c: int(new[c].isna().sum()) for c in colonnes}
    return new, envelope(
        "ok",
        f"Valeurs manquantes traitées ({strategie}) sur {', '.join(colonnes)}.",
        metrics={"avant": avant, "apres": apres, "lignes_avant": lignes_avant, "lignes_apres": len(new)},
    )

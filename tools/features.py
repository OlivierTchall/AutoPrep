"""engineer_features — outil de modification. 5 types fermés."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from tools.registry import envelope

_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_EXPR_OK = re.compile(r"^[\w\s+\-*/().]+$")
_OPS = {
    "<": lambda a, b: a < b, "<=": lambda a, b: a <= b, "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b, ">=": lambda a, b: a >= b, ">": lambda a, b: a > b,
}


def _resolve(df, token):
    if isinstance(token, str) and token in df.columns:
        return df[token]
    return token


def engineer_features(df, specification, justification):
    spec = specification
    typ, source, output, params = spec["type"], spec.get("source", []), spec["output"], spec.get("params", {})
    new = df.copy()
    added = []
    try:
        if typ == "formula":
            expr = params["expression"]
            if not _EXPR_OK.match(expr):
                return df, envelope("error", "Expression de formule non autorisée (caractères interdits).")
            for ident in set(_ID.findall(expr)):
                if ident not in new.columns:
                    return df, envelope("error", f"Identifiant '{ident}' de la formule n'est pas une colonne existante.")
            new[output] = new.eval(expr)
            added = [output]
        elif typ == "ratio":
            num, den = new[source[0]], new[source[1]]
            new[output] = np.where(den == 0, 0.0, num / den)
            added = [output]
        elif typ == "date_components":
            s = pd.to_datetime(new[source[0]], errors="coerce")
            mapping = {"annee": s.dt.year, "mois": s.dt.month, "jour": s.dt.day, "jour_semaine": s.dt.dayofweek}
            for comp in params.get("composants", ["annee", "mois"]):
                col = f"{output}_{comp}"
                new[col] = mapping[comp]
                added.append(col)
        elif typ == "binned":
            if "q" in params:
                new[output] = pd.qcut(new[source[0]], q=params["q"], labels=params.get("labels"), duplicates="drop")
            else:
                new[output] = pd.cut(new[source[0]], bins=params["bins"], labels=params.get("labels"))
            added = [output]
        elif typ == "flag":
            op = _OPS[params["operateur"]]
            left = _resolve(new, params["gauche"])
            right = _resolve(new, params["droite"])
            new[output] = op(left, right).astype(bool)
            added = [output]
        else:
            return df, envelope("error", f"Type de feature inconnu : {typ}.")
    except (KeyError, ValueError) as e:
        return df, envelope("error", f"Échec engineer_features : {e}")

    return new, envelope(
        "ok",
        f"Variable(s) dérivée(s) créée(s) : {', '.join(added)} (type {typ}).",
        metrics={"colonnes_ajoutees": added, "n_lignes": len(new),
                 "apercu": {c: new[c].head(5).tolist() for c in added}},
    )

"""query_dataframe — opérations paramétrées, jamais de code exécuté."""
from __future__ import annotations

import pandas as pd

from tools.registry import envelope
from tools.typing_ import parse_dates_tolerant

_CMP = {
    "<": lambda s, v: s < v, "<=": lambda s, v: s <= v, "==": lambda s, v: s == v,
    "!=": lambda s, v: s != v, ">=": lambda s, v: s >= v, ">": lambda s, v: s > v,
    "contains": lambda s, v: s.astype("string").str.contains(str(v), case=False, na=False),
}
_AGG = {"sum", "mean", "count", "min", "max", "median", "nunique"}
TOP_N_CAP = 20


def _apply_periode(df, periode):
    col = periode["colonne_date"]
    d = parse_dates_tolerant(df[col])
    out = df
    if periode.get("mois") is not None:
        out = out[d.dt.month == periode["mois"]]
        d = d[d.dt.month == periode["mois"]]
    if periode.get("annee") is not None:
        out = out[d.dt.year == periode["annee"]]
    return out


def query_dataframe(df, operation, parametres):
    p = parametres or {}
    try:
        if operation == "filter_count":
            s = df[p["colonne"]]
            n = int(_CMP[p["operateur"]](s, p["valeur"]).sum())
            return envelope("ok", f"{n} ligne(s) correspondent au filtre.", detail={"n": n})

        if operation == "groupby_agg":
            data = _apply_periode(df, p["periode"]) if p.get("periode") else df
            fn = p["fonction"]
            if fn not in _AGG:
                return envelope("error", f"Fonction d'agrégation inconnue : {fn}.")
            grouped = getattr(data.groupby(p["colonne_groupby"])[p["colonne_agg"]], fn)()
            grouped = grouped.sort_values(ascending=False)
            top = [grouped.index[0], round(float(grouped.iloc[0]), 4)] if len(grouped) else [None, None]
            return envelope(
                "ok",
                f"Agrégation {fn} de {p['colonne_agg']} par {p['colonne_groupby']}. En tête : {top[0]}.",
                detail={"resultats": {str(k): round(float(v), 4) for k, v in grouped.head(20).items()}, "top": top},
            )

        if operation == "describe_column":
            desc = df[p["colonne"]].describe()
            return envelope("ok", f"Statistiques descriptives de {p['colonne']}.",
                            detail={str(k): (float(v) if pd.api.types.is_number(v) else str(v)) for k, v in desc.items()})

        if operation == "top_n":
            n = int(p["n"])
            tronque = n > TOP_N_CAP
            n = min(n, TOP_N_CAP)
            asc = p.get("ordre", "desc") == "asc"
            rows = df.sort_values(p["colonne"], ascending=asc).head(n)
            msg = f"{n} première(s) ligne(s) par {p['colonne']}."
            if tronque:
                msg += f" (Tronqué à {TOP_N_CAP}.)"
            return envelope("ok", msg, detail={"lignes": rows.to_dict(orient="records"), "tronque": tronque})

        return envelope("error", f"Opération inconnue : {operation}.")
    except KeyError as e:
        return envelope("error", f"Paramètre ou colonne manquant : {e}.")

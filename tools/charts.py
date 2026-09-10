"""plot_chart — graphiques plotly."""
from __future__ import annotations

import plotly.express as px

from tools.registry import envelope

_BUILDERS = {
    "bar": lambda df, x, y: px.bar(df, x=x, y=y),
    "line": lambda df, x, y: px.line(df, x=x, y=y),
    "histogram": lambda df, x, y: px.histogram(df, x=x),
    "scatter": lambda df, x, y: px.scatter(df, x=x, y=y),
    "box": lambda df, x, y: px.box(df, x=x, y=y),
}


def plot_chart(df, type, x, y=None, titre=""):
    if type not in _BUILDERS:
        return envelope("error", f"Type de graphique non supporté : {type}.")
    for col in (c for c in (x, y) if c):
        if col not in df.columns:
            return envelope("error", f"Colonne inexistante : {col}.")
    fig = _BUILDERS[type](df, x, y)
    if titre:
        fig.update_layout(title=titre)
    return envelope("ok", f"Graphique {type} ({x}{' / ' + y if y else ''}) généré.",
                    detail={"figure": fig.to_dict()})

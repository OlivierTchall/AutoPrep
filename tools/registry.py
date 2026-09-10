"""Schémas d'arguments (pydantic), classification lecture/écriture, TOOL_REGISTRY, dispatcher."""
from __future__ import annotations

import importlib
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


def envelope(status: str, summary: str, metrics: dict | None = None, detail: Any = None) -> dict:
    return {"status": status, "summary": summary, "metrics": metrics or {}, "detail": detail}


# ---------- arg models: read tools ----------
class ProfileDatasetArgs(BaseModel):
    pass


class DetectOutliersArgs(BaseModel):
    colonnes: list[str] = Field(..., description="Colonnes numériques à analyser.")
    methode: Literal["iqr", "zscore"] = Field("iqr", description="Méthode de détection.")


class GenerateReportArgs(BaseModel):
    pass


class PlotChartArgs(BaseModel):
    type: Literal["bar", "line", "histogram", "scatter", "box"]
    x: str
    y: str | None = None
    titre: str = ""


class _Periode(BaseModel):
    colonne_date: str
    mois: int | None = None
    annee: int | None = None


class QueryDataframeArgs(BaseModel):
    operation: Literal["groupby_agg", "filter_count", "describe_column", "top_n"]
    parametres: dict = Field(default_factory=dict)


# ---------- arg models: write tools ----------
class FixColumnTypesArgs(BaseModel):
    colonnes: list[str]
    types_cibles: dict[str, Literal["entier", "decimal", "date", "texte", "categorie"]]
    justification: str


class HandleDuplicatesArgs(BaseModel):
    sous_ensemble_colonnes: list[str] | None = None
    justification: str


class HandleMissingValuesArgs(BaseModel):
    colonnes: list[str]
    strategie: Literal["drop_rows", "mean", "median", "mode", "constant"]
    valeur_constante: Any | None = None
    justification: str

    @model_validator(mode="after")
    def _constant_needs_value(self):
        if self.strategie == "constant" and self.valeur_constante is None:
            raise ValueError("valeur_constante est requise quand strategie == 'constant'")
        return self


class _BornesValides(BaseModel):
    min: float | None = None
    max: float | None = None


class TreatOutliersArgs(BaseModel):
    colonnes: list[str]
    action: Literal["cap", "remove_rows", "set_nan"]
    justification: str
    methode: Literal["iqr", "zscore"] = "iqr"
    bornes_valides: _BornesValides | None = Field(
        None, description="Plage métier valide ; si fournie, les lignes hors plage sont la cible (violation de règle métier).",
    )


class _FeatureSpec(BaseModel):
    type: Literal["date_components", "ratio", "binned", "flag", "formula"]
    source: list[str]
    output: str
    params: dict = Field(default_factory=dict)


class EngineerFeaturesArgs(BaseModel):
    specification: _FeatureSpec
    justification: str


# ---------- registry ----------
_META = {
    "profile_dataset": ("Profiler le jeu de données", [], ProfileDatasetArgs, False),
    "detect_outliers": ("Détecter les valeurs extrêmes", ["colonnes", "methode"], DetectOutliersArgs, False),
    "generate_report": ("Générer le rapport final", [], GenerateReportArgs, False),
    "plot_chart": ("Tracer un graphique", ["type", "x", "y"], PlotChartArgs, False),
    "query_dataframe": ("Interroger les données", ["operation", "parametres"], QueryDataframeArgs, False),
    "fix_column_types": ("Corriger le type d'une colonne", ["colonnes", "types_cibles"], FixColumnTypesArgs, True),
    "handle_duplicates": ("Supprimer les doublons", ["sous_ensemble_colonnes"], HandleDuplicatesArgs, True),
    "handle_missing_values": ("Traiter les valeurs manquantes", ["colonnes", "strategie", "valeur_constante"], HandleMissingValuesArgs, True),
    "treat_outliers": ("Traiter les valeurs extrêmes", ["colonnes", "action", "bornes_valides"], TreatOutliersArgs, True),
    "engineer_features": ("Créer une variable dérivée", ["specification"], EngineerFeaturesArgs, True),
}

TOOL_REGISTRY = {
    name: {"libelle": lib, "params_affiches": params, "args_model": model, "write": write}
    for name, (lib, params, model, write) in _META.items()
}
WRITE_TOOLS = {n for n, m in TOOL_REGISTRY.items() if m["write"]}
READ_TOOLS = {n for n, m in TOOL_REGISTRY.items() if not m["write"]}
LLM_TOOLS = [m["args_model"] for m in TOOL_REGISTRY.values()]

# pydantic model class name -> tool name, so ChatAnthropic tool_calls (which use the
# model's class name) map back to our registry keys.
MODEL_NAME_TO_TOOL = {m["args_model"].__name__: n for n, m in TOOL_REGISTRY.items()}


def validate_args(tool_name: str, args: dict) -> dict:
    model = TOOL_REGISTRY[tool_name]["args_model"]
    return model.model_validate(args or {}).model_dump()


_DISPATCH_TABLE = {
    "profile_dataset": ("tools.profiling", "profile_dataset"),
    "detect_outliers": ("tools.outliers", "detect_outliers"),
    "plot_chart": ("tools.charts", "plot_chart"),
    "query_dataframe": ("tools.query", "query_dataframe"),
    "fix_column_types": ("tools.typing_", "fix_column_types"),
    "handle_duplicates": ("tools.cleaning", "handle_duplicates"),
    "handle_missing_values": ("tools.cleaning", "handle_missing_values"),
    "treat_outliers": ("tools.outliers", "treat_outliers"),
    "engineer_features": ("tools.features", "engineer_features"),
}


def dispatch(tool_name: str, df, args: dict):
    """Execute a tool. Returns (new_df_or_None, envelope). generate_report is handled by the graph, not here."""
    clean = validate_args(tool_name, args)
    mod_name, fn_name = _DISPATCH_TABLE[tool_name]
    fn = getattr(importlib.import_module(mod_name), fn_name)
    result = fn(df, **clean)
    if tool_name in WRITE_TOOLS:
        new_df, env = result
        return new_df, env
    return None, result

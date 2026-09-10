import numpy as np
import pandas as pd
from tools.outliers import detect_outliers, treat_outliers


def test_detect_outliers_iqr():
    df = pd.DataFrame({"x": [10, 11, 12, 10, 11, 9, 10, 500]})
    out = detect_outliers(df, colonnes=["x"], methode="iqr")
    assert out["status"] == "ok"
    assert out["detail"]["x"]["n_outliers"] == 1


def test_treat_outliers_business_rule_remove(raw_df):
    new, env = treat_outliers(raw_df, colonnes=["quantite"], action="remove_rows",
                              justification="Quantités négatives = violation de règle métier (profilage: min < 0).",
                              bornes_valides={"min": 0, "max": None})
    assert env["status"] == "ok"
    assert new["quantite"].min() >= 0
    assert env["metrics"]["lignes_apres"] < env["metrics"]["lignes_avant"]


def test_treat_outliers_cap_with_bounds():
    df = pd.DataFrame({"q": [-3, -1, 2, 5, 8]})
    new, env = treat_outliers(df, colonnes=["q"], action="cap", justification="x",
                              bornes_valides={"min": 0, "max": None})
    assert new["q"].min() == 0

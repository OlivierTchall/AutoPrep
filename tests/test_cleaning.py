import numpy as np
import pandas as pd
from tools.cleaning import handle_duplicates, handle_missing_values


def test_handle_duplicates_removes_all(raw_df):
    new, env = handle_duplicates(raw_df, sous_ensemble_colonnes=None, justification="2% de doublons au profilage.")
    assert env["status"] == "ok"
    assert new.duplicated().sum() == 0
    assert env["metrics"]["supprimees"] == env["metrics"]["avant"] - env["metrics"]["apres"]


def test_missing_mode_fills_categorie(raw_df):
    new, env = handle_missing_values(raw_df, colonnes=["categorie", "mode_paiement"], strategie="mode",
                                     valeur_constante=None, justification="6% de manquants.")
    assert env["status"] == "ok"
    assert new["categorie"].isna().sum() == 0
    assert new["mode_paiement"].isna().sum() == 0


def test_missing_constant():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    new, env = handle_missing_values(df, colonnes=["a"], strategie="constant",
                                     valeur_constante=0, justification="x")
    assert new["a"].tolist() == [1.0, 0.0, 3.0]


def test_missing_mean_on_text_is_error(raw_df):
    new, env = handle_missing_values(raw_df, colonnes=["categorie"], strategie="mean",
                                     valeur_constante=None, justification="x")
    assert env["status"] == "error"

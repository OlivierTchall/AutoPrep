import pandas as pd
from tools.features import engineer_features


def test_formula_recomputes_montant(raw_df):
    new, env = engineer_features(
        raw_df,
        specification={"type": "formula", "source": ["quantite", "prix_unitaire"],
                       "output": "montant_total_recalc",
                       "params": {"expression": "quantite * prix_unitaire"}},
        justification="montant_total incohérent avec quantite*prix_unitaire sur une partie des lignes.",
    )
    assert env["status"] == "ok"
    assert "montant_total_recalc" in new.columns
    assert (new["montant_total_recalc"] == (raw_df["quantite"] * raw_df["prix_unitaire"])).all()


def test_flag_ecart(raw_df):
    df = raw_df.assign(montant_total_recalc=raw_df["quantite"] * raw_df["prix_unitaire"])
    new, env = engineer_features(
        df,
        specification={"type": "flag", "source": ["montant_total", "montant_total_recalc"],
                       "output": "ecart_montant",
                       "params": {"gauche": "montant_total", "operateur": "!=", "droite": "montant_total_recalc"}},
        justification="Marquer les lignes où le montant déclaré diffère du recalcul.",
    )
    assert new["ecart_montant"].dtype == bool
    assert new["ecart_montant"].sum() >= 20


def test_formula_rejects_non_column_identifier(raw_df):
    new, env = engineer_features(
        raw_df,
        specification={"type": "formula", "source": ["quantite"], "output": "x",
                       "params": {"expression": "quantite * __import__"}},
        justification="x",
    )
    assert env["status"] == "error"

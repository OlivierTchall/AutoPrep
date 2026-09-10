import pandas as pd
from data.generate_synthetic import generate_synthetic


def test_reproducible():
    a = generate_synthetic(1000)
    b = generate_synthetic(1000)
    pd.testing.assert_frame_equal(a, b)


def test_shape_and_columns():
    df = generate_synthetic(1000)
    assert len(df) == 1000
    assert list(df.columns) == [
        "order_id", "date_commande", "client", "produit", "categorie",
        "quantite", "prix_unitaire", "montant_total", "region", "mode_paiement",
    ]


def test_injected_anomalies():
    df = generate_synthetic(1000)
    # dates kept as raw strings, two formats mixed
    assert df["date_commande"].dtype == object
    slash = df["date_commande"].str.contains("/").mean()
    assert 0.10 < slash < 0.25
    # missing values ~6% on categorie and mode_paiement
    assert 0.03 < df["categorie"].isna().mean() < 0.09
    assert 0.03 < df["mode_paiement"].isna().mean() < 0.09
    # ~2% duplicated rows
    assert 0.01 <= df.duplicated().mean() <= 0.04
    # business-rule violation: some negative quantities
    assert (df["quantite"] < 0).sum() >= 3
    # montant_total inconsistent with quantite*prix_unitaire on some rows
    prod = (df["quantite"] * df["prix_unitaire"]).round(2)
    assert (prod != df["montant_total"].round(2)).sum() >= 20

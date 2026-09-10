from tools.query import query_dataframe


def test_top_product_in_march(raw_df):
    df = raw_df.copy()
    df = df[df["quantite"] > 0]
    out = query_dataframe(df, "groupby_agg", {
        "colonne_groupby": "produit", "colonne_agg": "quantite", "fonction": "sum",
        "periode": {"colonne_date": "date_commande", "mois": 3},
    })
    assert out["status"] == "ok"
    assert out["detail"]["top"][0] in df["produit"].unique()


def test_filter_count(raw_df):
    out = query_dataframe(raw_df, "filter_count", {"colonne": "quantite", "operateur": "<", "valeur": 0})
    assert out["detail"]["n"] == int((raw_df["quantite"] < 0).sum())


def test_top_n_caps_at_20(raw_df):
    out = query_dataframe(raw_df, "top_n", {"colonne": "montant_total", "n": 50, "ordre": "desc"})
    assert out["detail"]["tronque"] is True
    assert len(out["detail"]["lignes"]) == 20

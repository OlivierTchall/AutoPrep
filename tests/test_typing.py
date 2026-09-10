import pandas as pd
from tools.typing_ import fix_column_types, parse_dates_tolerant


def test_parse_mixed_date_formats():
    s = pd.Series(["2024-03-15", "01/04/2024", "31/12/2023", "pas une date"])
    out = parse_dates_tolerant(s)
    assert out.dtype == "datetime64[ns]"
    assert out.isna().sum() == 1
    assert out.iloc[1] == pd.Timestamp("2024-04-01")


def test_fix_column_types_on_synthetic(raw_df):
    new, env = fix_column_types(
        raw_df, colonnes=["date_commande", "quantite"],
        types_cibles={"date_commande": "date", "quantite": "entier"},
        justification="Le profilage montre date_commande en texte et deux formats mélangés.",
    )
    assert env["status"] == "ok"
    assert new["date_commande"].dtype == "datetime64[ns]"
    assert str(new["quantite"].dtype).startswith("int") or str(new["quantite"].dtype).startswith("Int")
    assert env["metrics"]["apres"]["date_commande"] == "datetime64[ns]"

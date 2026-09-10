from tools.profiling import profile_dataset


def test_profile_flags_synthetic_anomalies(raw_df):
    out = profile_dataset(raw_df)
    assert out["status"] == "ok"
    d = out["detail"]
    assert d["n_lignes"] == len(raw_df)
    assert d["n_doublons"] >= 1
    assert d["valeurs_manquantes"]["categorie"]["n"] > 0
    # min/max exposed per numeric column, and negative quantite visible
    assert d["numeriques"]["quantite"]["min"] < 0
    assert "quantite" in out["summary"].lower() or "négative" in out["summary"].lower()

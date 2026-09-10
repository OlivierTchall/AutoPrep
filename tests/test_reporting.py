from tools.reporting import generate_report


def test_report_has_all_sections(raw_df, tmp_path):
    df2 = raw_df.drop_duplicates()
    log = [
        {"outil": "profile_dataset", "parametres": {}, "justification": "", "metriques": {"n_lignes": len(raw_df)},
         "timestamp": "2026-09-10T10:00:00", "type": "lecture", "origine": "auto"},
        {"outil": "handle_duplicates", "parametres": {"sous_ensemble_colonnes": None},
         "justification": "2% de doublons", "metriques": {"supprimees": 20},
         "timestamp": "2026-09-10T10:01:00", "type": "modification", "origine": "humain_validee"},
    ]
    out = generate_report(raw_df, df2, log, path=str(tmp_path / "rapport.md"))
    md = out["detail"]["markdown"]
    for h in ["# Rapport de valorisation AutoPrep", "## Résumé exécutif", "## État initial",
              "## Actions effectuées", "## État final", "## Variables créées"]:
        assert h in md
    assert "Lecture (auto)" in md
    assert "Modification (validée)" in md
    assert (tmp_path / "rapport.md").exists()

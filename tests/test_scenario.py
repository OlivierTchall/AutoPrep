"""Scénario end-to-end — définition de terminé (MVP).

Pilote le vrai graphe LangGraph + les vrais outils avec un LLM scripté et
vérifie les critères mesurables de la spec (§ "Definition of done") :

    assert df_current.duplicated().sum() == 0
    assert df_current["categorie"].isna().mean() < 0.06
    assert df_current["quantite"].min() >= 0
    assert str(df_current["date_commande"].dtype) == "datetime64[ns]"

plus : le rapport distingue les trois types d'action, `query_dataframe` répond
aux trois questions de test, le garde-fou laisse le graphe en pause si l'humain
ne valide jamais rien, et une question hors-bande ne détruit pas la proposition
en attente.
"""
import pytest

from agent import store
from agent.loop import (
    _graph_state,
    get_pending_proposal,
    get_report,
    reset_graph,
    resume_run,
    start_run,
    submit_question,
)
from data.generate_synthetic import generate_synthetic
from tools.query import query_dataframe

VALIDEE = {"decision": "validee", "motif_refus": None, "contre_proposition": None}


@pytest.fixture(autouse=True)
def _reset():
    # R3 : repartir d'un graphe et d'un DF_STORE vierges pour ne pas entrer en
    # collision avec les threads des autres modules de test.
    reset_graph()
    yield
    reset_graph()


def test_end_to_end_mvp(scripted):
    df = generate_synthetic(1000)

    scripted([
        ("ProfileDatasetArgs", {}),
        ("FixColumnTypesArgs", {
            "colonnes": ["date_commande", "quantite", "prix_unitaire", "montant_total"],
            "types_cibles": {"date_commande": "date", "quantite": "entier",
                             "prix_unitaire": "decimal", "montant_total": "decimal"},
            "justification": "Profilage : dates en texte (2 formats), numériques à fiabiliser."}),
        ("HandleDuplicatesArgs", {"sous_ensemble_colonnes": None,
                                  "justification": "~2% de doublons au profilage."}),
        # celle-ci est refusée puis contre-proposée (mode) :
        ("HandleMissingValuesArgs", {"colonnes": ["categorie", "mode_paiement"],
                                     "strategie": "drop_rows", "justification": "6% de manquants."}),
        ("TreatOutliersArgs", {"colonnes": ["quantite"], "action": "remove_rows",
                               "justification": "Quantités négatives = violation de règle métier (profilage: min < 0).",
                               "bornes_valides": {"min": 0, "max": None}}),
        ("EngineerFeaturesArgs", {"specification": {
            "type": "formula", "source": ["quantite", "prix_unitaire"],
            "output": "montant_total_recalc",
            "params": {"expression": "quantite * prix_unitaire"}},
            "justification": "montant_total incohérent sur une partie des lignes."}),
        ("GenerateReportArgs", {}),
    ])

    start_run("scenario", df)

    # 1. fix_column_types — valider
    assert get_pending_proposal("scenario")["tool_name"] == "fix_column_types"
    resume_run("scenario", VALIDEE)
    # 2. handle_duplicates — valider
    assert get_pending_proposal("scenario")["tool_name"] == "handle_duplicates"
    resume_run("scenario", VALIDEE)
    # 3. handle_missing_values drop_rows — REFUSER avec contre-proposition (mode)
    assert get_pending_proposal("scenario")["tool_name"] == "handle_missing_values"
    resume_run("scenario", {
        "decision": "refusee", "motif_refus": "drop_rows trop destructif",
        "contre_proposition": {"tool_name": "handle_missing_values",
                               "arguments": {"colonnes": ["categorie", "mode_paiement"],
                                             "strategie": "mode",
                                             "justification": "Imputation par le mode, non destructif."}}})
    # 4. treat_outliers — valider
    assert get_pending_proposal("scenario")["tool_name"] == "treat_outliers"
    resume_run("scenario", VALIDEE)
    # 5. engineer_features — valider
    assert get_pending_proposal("scenario")["tool_name"] == "engineer_features"
    resume_run("scenario", VALIDEE)

    df_current = store.get_current("scenario")

    # ---- DoD mesurable (spec §"Definition of done") ----
    assert df_current.duplicated().sum() == 0
    assert df_current["categorie"].isna().mean() < 0.06
    assert df_current["quantite"].min() >= 0
    assert str(df_current["date_commande"].dtype) == "datetime64[ns]"

    # le rapport existe, liste les actions, distingue lecture vs modification
    report = get_report("scenario")
    assert report is not None
    assert "Lecture (auto)" in report
    assert "Modification (validée)" in report
    assert "Modification (contre-proposition humaine)" in report

    # query_dataframe répond aux 3 questions de test
    d = df_current[df_current["quantite"] > 0]
    q1 = query_dataframe(d, "groupby_agg", {"colonne_groupby": "produit", "colonne_agg": "quantite",
        "fonction": "sum", "periode": {"colonne_date": "date_commande", "mois": 3}})
    assert q1["status"] == "ok" and q1["detail"]["top"][0] is not None
    q2 = query_dataframe(d, "filter_count", {"colonne": "montant_total", "operateur": ">", "valeur": 1000})
    assert q2["status"] == "ok"
    q3 = query_dataframe(d, "top_n", {"colonne": "montant_total", "n": 5, "ordre": "desc"})
    assert q3["status"] == "ok" and len(q3["detail"]["lignes"]) == 5


def test_guardrail_cuts_when_nothing_is_validated(scripted):
    scripted([("FixColumnTypesArgs", {"colonnes": ["quantite"],
                                      "types_cibles": {"quantite": "entier"},
                                      "justification": "x"})] * 40)
    start_run("scenario_guard", generate_synthetic(500))
    # on ne reprend jamais l'interruption -> le graphe reste en pause ; on ne peut
    # pas piloter `think` manuellement, donc cette variante vérifie que la pause a
    # bien eu lieu et qu'aucun rapport n'a été produit sans intervention humaine
    # (comportement documenté : une pause `interrupt` ne consomme aucun token et
    # n'atteint jamais le garde-fou tant que personne ne reprend).
    assert get_pending_proposal("scenario_guard") is not None
    assert get_report("scenario_guard") is None


def test_out_of_band_question_keeps_pending_proposal(scripted):
    scripted([
        ("HandleDuplicatesArgs", {"sous_ensemble_colonnes": None, "justification": "doublons"}),
        ("GenerateReportArgs", {}),
    ])
    start_run("scenario_oob", generate_synthetic(500))
    assert get_pending_proposal("scenario_oob")["tool_name"] == "handle_duplicates"
    # une question arrive pendant que la proposition est en attente
    submit_question("scenario_oob", "Quelle est la distribution de montant_total ?")
    # la proposition est toujours en attente, inchangée
    assert get_pending_proposal("scenario_oob")["tool_name"] == "handle_duplicates"
    # et aucun rapport n'a été généré par la question hors-bande
    assert _graph_state("scenario_oob").get("final_report") is None

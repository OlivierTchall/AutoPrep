"""Tests du graphe LangGraph : lecture directe, pause HITL, garde-fou d'itérations, refus motivé."""
import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from agent import store
from agent.loop import (
    _graph_state,
    get_pending_proposal,
    get_report,
    reset_graph,
    resume_run,
    start_run,
)


@pytest.fixture(autouse=True)
def _reset():
    # R3 : repartir d'un graphe et d'un DF_STORE vierges avant chaque test
    # (sinon le checkpointer InMemorySaver et DF_STORE fuient d'un test à l'autre).
    reset_graph()
    yield
    reset_graph()


def test_read_tool_runs_without_interrupt(scripted, raw_df):
    scripted([("ProfileDatasetArgs", {}), ("GenerateReportArgs", {})])
    start_run("tr1", raw_df)
    assert get_pending_proposal("tr1") is None
    assert get_report("tr1") is not None


def test_write_tool_pauses_then_resumes(scripted, raw_df):
    scripted([
        ("HandleDuplicatesArgs", {"sous_ensemble_colonnes": None, "justification": "2% doublons"}),
        ("GenerateReportArgs", {}),
    ])
    start_run("tw1", raw_df)
    prop = get_pending_proposal("tw1")
    assert prop["tool_name"] == "handle_duplicates"
    resume_run("tw1", {"decision": "validee", "motif_refus": None, "contre_proposition": None})
    assert store.get_current("tw1").duplicated().sum() == 0
    assert get_pending_proposal("tw1") is None
    assert get_report("tw1") is not None


def test_guardrail_forces_report_at_15(scripted, raw_df):
    # Le LLM propose indéfiniment un outil de lecture ; le garde-fou doit couper.
    scripted([("ProfileDatasetArgs", {})] * 40)
    start_run("tg1", raw_df)
    rep = get_report("tg1")
    assert rep is not None
    assert "Itération maximale atteinte (15)" in rep
    assert _graph_state("tg1")["iteration_count"] >= 15
    assert _graph_state("tg1")["task_done"] is True


def test_start_run_seeds_a_human_message(scripted, raw_df):
    # C1 : `think` doit toujours voir >= 1 message non-système (sinon Anthropic 400).
    scripted([("ProfileDatasetArgs", {}), ("GenerateReportArgs", {})])
    start_run("tc1", raw_df)
    msgs = _graph_state("tc1")["messages"]
    assert msgs
    assert isinstance(msgs[0], HumanMessage)


def test_tool_exception_yields_error_toolmessage_and_run_completes(scripted, raw_df, monkeypatch):
    # I2 : dispatch qui lève -> enveloppe d'erreur, pas de crash, run terminé.
    import agent.loop as loop_mod

    real_dispatch = loop_mod.dispatch
    calls = {"n": 0}

    def flaky_dispatch(name, df, args):
        calls["n"] += 1
        if name == "profile_dataset":
            raise RuntimeError("boom")
        return real_dispatch(name, df, args)

    monkeypatch.setattr(loop_mod, "dispatch", flaky_dispatch)
    scripted([("ProfileDatasetArgs", {}), ("GenerateReportArgs", {})])
    start_run("ti2", raw_df)
    assert get_report("ti2") is not None
    msgs = _graph_state("ti2")["messages"]
    assert any(isinstance(m, ToolMessage) and '"error"' in m.content for m in msgs)


def test_text_only_turn_is_nudged_back_on_track(scripted, raw_df):
    # I3 : un tour LLM sans appel d'outil ne doit pas terminer le run à vide.
    scripted(["Je réfléchis...", ("ProfileDatasetArgs", {}), ("GenerateReportArgs", {})])
    start_run("ti3", raw_df)
    assert get_report("ti3") is not None
    assert _graph_state("ti3")["iteration_count"] >= 2


def test_non_object_counter_proposal_does_not_crash(scripted, raw_df):
    # I4 : contre-proposition qui n'est pas un objet -> refus simple, pas de crash.
    scripted([
        ("HandleDuplicatesArgs", {"sous_ensemble_colonnes": None, "justification": "x"}),
        ("GenerateReportArgs", {}),
    ])
    start_run("ti4", raw_df)
    assert get_pending_proposal("ti4")["tool_name"] == "handle_duplicates"
    resume_run("ti4", {"decision": "refusee", "motif_refus": None, "contre_proposition": []})
    # le graphe a repris (think) et poursuivi jusqu'au rapport
    assert get_report("ti4") is not None


def test_parallel_tool_calls_do_not_orphan_tool_use_ids(scripted, raw_df):
    # Bug live constaté avec l'API réelle (Claude Haiku 4.5) : le modèle a renvoyé
    # plusieurs tool_calls dans un même tour ; seul tool_calls[0] était traité et les
    # autres restaient sans tool_result -> Anthropic 400 au tour suivant. `call_llm`
    # désactive maintenant `parallel_tool_calls`, et `act` répond quand même à tout
    # tool_call excédentaire par un tool_result "skipped" en défense en profondeur.
    scripted([
        [("ProfileDatasetArgs", {}), ("DetectOutliersArgs", {"colonnes": ["quantite"], "methode": "iqr"})],
        ("GenerateReportArgs", {}),
    ])
    start_run("tpar1", raw_df)
    msgs = _graph_state("tpar1")["messages"]
    assert any(isinstance(m, ToolMessage) and '"skipped"' in m.content for m in msgs)
    # le premier tool_call (profile_dataset) a bien été exécuté, la boucle a continué
    assert get_report("tpar1") is not None


def test_parallel_tool_calls_on_a_refused_write_tool_still_stub_the_rest(scripted, raw_df):
    # Même défense, mais quand tool_calls[0] est un outil d'écriture refusé sans
    # contre-proposition : c'est `human_gate`, pas `act`, qui répond au premier appel.
    scripted([
        [
            ("HandleDuplicatesArgs", {"sous_ensemble_colonnes": None, "justification": "x"}),
            ("DetectOutliersArgs", {"colonnes": ["quantite"], "methode": "iqr"}),
        ],
        ("GenerateReportArgs", {}),
    ])
    start_run("tpar2", raw_df)
    assert get_pending_proposal("tpar2")["tool_name"] == "handle_duplicates"
    resume_run("tpar2", {"decision": "refusee", "motif_refus": "x", "contre_proposition": None})
    msgs = _graph_state("tpar2")["messages"]
    assert any(isinstance(m, ToolMessage) and '"skipped"' in m.content for m in msgs)
    assert get_report("tpar2") is not None


def test_refusal_with_motif_goes_back_to_think(scripted, raw_df):
    scripted([
        ("HandleMissingValuesArgs", {"colonnes": ["categorie"], "strategie": "drop_rows", "justification": "x"}),
        ("HandleMissingValuesArgs", {"colonnes": ["categorie"], "strategie": "mode", "justification": "moins destructif"}),
        ("GenerateReportArgs", {}),
    ])
    start_run("trf1", raw_df)
    resume_run("trf1", {"decision": "refusee", "motif_refus": "drop trop destructif", "contre_proposition": None})
    prop = get_pending_proposal("trf1")
    assert prop["tool_name"] == "handle_missing_values"
    assert prop["arguments"]["strategie"] == "mode"

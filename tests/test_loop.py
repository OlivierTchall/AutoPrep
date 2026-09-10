"""Tests du graphe LangGraph : lecture directe, pause HITL, garde-fou d'itérations, refus motivé."""
import pytest

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

"""Schéma de l'état du graphe LangGraph."""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

import config

CHECKLIST_KEYS = ["profilage", "typage", "doublons", "valeurs_manquantes", "outliers", "features"]

TOOL_TO_CHECKLIST = {
    "profile_dataset": "profilage",
    "fix_column_types": "typage",
    "handle_duplicates": "doublons",
    "handle_missing_values": "valeurs_manquantes",
    "treat_outliers": "outliers",
    "engineer_features": "features",
}


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    thread_id: str
    iteration_count: int
    max_iterations: int
    checklist: dict
    pending_proposal: dict | None
    human_decision: str | None
    motif_refus: str | None
    human_proposed_action: dict | None
    actions_log: Annotated[list, operator.add]
    task_done: bool
    final_report: str | None


def initial_state(thread_id: str) -> AgentState:
    return AgentState(
        messages=[],
        thread_id=thread_id,
        iteration_count=0,
        max_iterations=config.MAX_ITERATIONS,
        checklist={k: False for k in CHECKLIST_KEYS},
        pending_proposal=None,
        human_decision=None,
        motif_refus=None,
        human_proposed_action=None,
        actions_log=[],
        task_done=False,
        final_report=None,
    )

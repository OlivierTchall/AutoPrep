"""Interface Streamlit d'AutoPrep Agent.

Couche mince : `st.session_state` ne contient QUE `thread_id` et le graphe compilé.
Tout le reste (messages, checklist, actions_log, proposition en attente, rapport)
vit dans l'état LangGraph et n'est lu/écrit que via les helpers de `agent.loop`.
"""
from __future__ import annotations

import json
from uuid import uuid4

import pandas as pd
import streamlit as st

from agent.loop import (
    build_graph,
    get_pending_proposal,
    get_report,
    resume_run,
    start_run,
    submit_question,
    _graph_state,
)
from agent.state import CHECKLIST_KEYS
from agent.upload import read_uploaded_csv, validate_dataframe
from tools.registry import TOOL_REGISTRY


def _init_session() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid4())
    if "graph" not in st.session_state:
        st.session_state.graph = build_graph()


def _render_checklist(checklist: dict) -> None:
    st.subheader("Checklist")
    for cle in CHECKLIST_KEYS:
        coche = "✅" if checklist.get(cle) else "⬜"
        st.write(f"{coche} {cle}")


def _render_actions_log(actions_log: list) -> None:
    st.subheader("Journal des actions")
    if not actions_log:
        st.caption("Aucune action pour l'instant.")
        return
    st.table(pd.DataFrame(actions_log))


def _render_validation_panel(thread_id: str, proposal: dict) -> None:
    name = proposal["tool_name"]
    args = proposal.get("arguments", {}) or {}
    meta = TOOL_REGISTRY.get(name, {"libelle": name, "params_affiches": []})

    st.subheader("Validation requise")
    st.markdown(f"**Quoi :** {meta['libelle']}")

    comment = {p: args.get(p) for p in meta["params_affiches"]}
    st.markdown("**Comment :**")
    st.json(comment)

    st.markdown(f"**Pourquoi :** {args.get('justification', '(non précisé)')}")

    col_ok, col_ko = st.columns(2)
    if col_ok.button("Valider", type="primary"):
        resume_run(
            thread_id,
            {"decision": "validee", "motif_refus": None, "contre_proposition": None},
        )
        st.rerun()

    with col_ko.expander("Refuser"):
        motif = st.text_area("Motif du refus")
        contre_txt = st.text_area(
            "Contre-proposition (JSON optionnel : "
            '{"tool_name": "...", "arguments": {...}})',
        )
        if st.button("Confirmer le refus"):
            contre = None
            contre_txt = (contre_txt or "").strip()
            if contre_txt:
                try:
                    contre = json.loads(contre_txt)
                    if not isinstance(contre, dict):
                        st.error("La contre-proposition doit être un objet JSON.")
                        contre = None
                except json.JSONDecodeError:
                    st.error("Contre-proposition ignorée : JSON invalide.")
                    contre = None
            resume_run(
                thread_id,
                {
                    "decision": "refusee",
                    "motif_refus": motif or None,
                    "contre_proposition": contre,
                },
            )
            st.rerun()


def _render_report(report_md: str) -> None:
    st.subheader("Rapport final")
    st.markdown(report_md)
    st.download_button(
        "Télécharger rapport.md",
        data=report_md,
        file_name="rapport.md",
        mime="text/markdown",
    )


def main() -> None:
    st.set_page_config(page_title="AutoPrep Agent", layout="wide")
    _init_session()
    thread_id = st.session_state.thread_id

    st.title("AutoPrep Agent")
    st.caption("Nettoyage assisté de jeux de données de ventes PME (human-in-the-loop).")

    state = _graph_state(thread_id)

    fichier = st.file_uploader("Jeu de données (CSV)", type=["csv"])
    if fichier is not None and not state:
        try:
            df = read_uploaded_csv(fichier.getvalue())
            validate_dataframe(df)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        start_run(thread_id, df)
        state = _graph_state(thread_id)

    if not state:
        st.info("Chargez un fichier CSV pour démarrer l'agent.")
        return

    col_gauche, col_droite = st.columns([2, 1])

    with col_droite:
        _render_checklist(state.get("checklist", {}))
        st.metric("Itérations", state.get("iteration_count", 0))

    with col_gauche:
        proposal = get_pending_proposal(thread_id)
        report_md = get_report(thread_id)

        if proposal is not None:
            _render_validation_panel(thread_id, proposal)
        elif report_md is not None:
            _render_report(report_md)
        else:
            st.info("L'agent travaille… (aucune action en attente de validation).")

        _render_actions_log(state.get("actions_log", []))

    question = st.chat_input("Poser une question sur les données")
    if question:
        st.session_state.last_answer = submit_question(thread_id, question)
        st.rerun()
    if st.session_state.get("last_answer"):
        st.chat_message("assistant").write(st.session_state.last_answer)


if __name__ == "__main__":
    main()

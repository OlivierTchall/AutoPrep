"""Graphe LangGraph : think / act / human_gate / force_report / report.

Boucle ReAct avec human-in-the-loop : les 5 outils de WRITE_TOOLS passent par
`interrupt()` dans `human_gate` avant exécution ; les outils de lecture vont
directement à `act`. Garde-fou d'itérations : `iteration_count` n'est incrémenté
que dans `think`, jamais pendant la pause d'interruption.

Les dataframes ne transitent jamais par l'état du graphe ni par `messages` :
`act` lit `store.get_current(tid)`, appelle `dispatch`, réécrit via
`store.set_current(tid, new_df)`. Le `ToolMessage` ajouté à `messages` ne porte
que `{"status", "summary", "metrics"}`.
"""
from __future__ import annotations

import json
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import ValidationError

from agent import store
from agent.llm_client import call_llm
from agent.prompts import SYSTEM_PROMPT
from agent.state import AgentState, TOOL_TO_CHECKLIST, initial_state
from tools.registry import (
    LLM_TOOLS,
    MODEL_NAME_TO_TOOL,
    WRITE_TOOLS,
    dispatch,
    validate_args,
)
from tools.reporting import generate_report

_CAP_MESSAGE = "Itération maximale atteinte (15) — je génère le rapport avec l'état actuel."
_NUDGE_MESSAGE = (
    "Tu n'as appelé aucun outil. Repasse la checklist de sortie, puis appelle "
    "l'outil suivant approprié, ou generate_report si tout est validé."
)
_RECURSION_LIMIT = 100

_GRAPH = None

# Questions hors-bande posées pendant une pause `interrupt()` : `submit_question`
# les empile ici (hors état du graphe, pour ne pas toucher au snapshot en pause) ;
# `think` les draine et les rejoue dans `messages` au prochain cycle.
_PENDING_QA: dict[str, list[tuple[str, str]]] = {}


def cfg(thread_id: str) -> dict:
    """Config LangGraph pour un thread. `recursion_limit` élevé car la boucle
    think/act peut légitimement enchaîner jusqu'à ~15 itérations * plusieurs nœuds."""
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": _RECURSION_LIMIT}


# ---------------------------------------------------------------- helpers


def _pending_tool_call(state: AgentState):
    """(name, args, id) résolus depuis l'AIMessage-avec-tool_calls le plus récent.

    À l'entrée de `human_gate` comme de `act`, `messages[-1]` est toujours
    l'AIMessage produit par `think` (les branches validee / contre-proposition de
    `human_gate` n'ajoutent aucun message)."""
    tc = state["messages"][-1].tool_calls[0]
    name = MODEL_NAME_TO_TOOL.get(tc["name"], tc["name"])
    return name, tc["args"], tc["id"]


def _extra_tool_call_stubs(state: AgentState) -> list[ToolMessage]:
    """Défense en profondeur : si le modèle renvoie plusieurs `tool_calls` dans un même
    tour (parallel tool use — désactivé via `parallel_tool_calls=False` dans `call_llm`,
    mais on ne dépend pas uniquement de ce flag), seul `tool_calls[0]` est traité par ce
    cycle. Les autres reçoivent immédiatement un `tool_result` stub : sans ça, Anthropic
    rejette le prochain appel ('tool_use ids were found without tool_result blocks')."""
    extras = state["messages"][-1].tool_calls[1:]
    if not extras:
        return []
    return [
        ToolMessage(
            tool_call_id=tc["id"],
            content=json.dumps(
                {
                    "status": "skipped",
                    "summary": "Non exécuté : un seul outil est traité par cycle de raisonnement.",
                },
                ensure_ascii=False,
            ),
        )
        for tc in extras
    ]


def _answer_question(df, question: str) -> str:
    """Réponse best-effort, lecture seule, à une question hors-bande.

    MVP : `describe_column` sur une colonne dont le nom apparaît dans la question,
    sinon on liste les colonnes disponibles.
    TODO (roadmap) : mapping question -> opération plus fin, ou appel LLM dédié."""
    cols = list(df.columns)
    q = (question or "").lower()
    hit = next((c for c in cols if c.lower() in q), None)
    if hit is not None:
        try:
            _, env = dispatch(
                "query_dataframe",
                df,
                {"operation": "describe_column", "parametres": {"colonne": hit}},
            )
            if env["status"] == "ok":
                return f"{env['summary']} Détail : {env.get('detail')}"
        except Exception:
            pass
    return "Voici les colonnes disponibles : " + ", ".join(cols) + "."


# ---------------------------------------------------------------- nodes


def think(state: AgentState) -> dict:
    # Garde-fou : plafond atteint -> message verbatim, ni appel LLM ni incrément.
    if state["iteration_count"] >= state["max_iterations"]:
        return {"messages": [AIMessage(content=_CAP_MESSAGE)]}

    out: dict = {"iteration_count": state["iteration_count"] + 1}
    extra: list = []

    # Drainer les questions hors-bande posées pendant une pause (via `submit_question`,
    # hors graphe) : on les rejoue comme paires Human/AI dans l'historique, sans avoir
    # touché à la proposition en attente ni à l'interruption.
    for q, a in _PENDING_QA.pop(state["thread_id"], []):
        extra.append(HumanMessage(content=q))
        extra.append(AIMessage(content=a))

    response = call_llm(
        [SystemMessage(content=SYSTEM_PROMPT), *state["messages"], *extra], LLM_TOOLS
    )

    # Pré-stage de la proposition d'écriture DANS l'état, avant toute pause :
    # `pending_proposal` vit dans l'état (et non seulement dans la valeur de
    # l'interruption) pour que `get_pending_proposal` reste fiable pendant la
    # pause, même si un `update_state` externe purgeait `snap.interrupts` (R1).
    if getattr(response, "tool_calls", None):
        tc = response.tool_calls[0]
        name = MODEL_NAME_TO_TOOL.get(tc["name"], tc["name"])
        if name in WRITE_TOOLS:
            out["pending_proposal"] = {"tool_name": name, "arguments": tc["args"]}

    out["messages"] = [*extra, response]
    return out


def route_after_think(state: AgentState) -> str:
    last = state["messages"][-1]
    has_calls = bool(getattr(last, "tool_calls", None))
    if state["iteration_count"] >= state["max_iterations"] and not has_calls:
        return "force_report"
    if not has_calls:
        return "nudge"
    name, _, _ = _pending_tool_call(state)
    if name == "generate_report":
        return "report"
    if name in WRITE_TOOLS:
        return "human_gate"
    return "act"


def human_gate(state: AgentState) -> dict:
    name, args, call_id = _pending_tool_call(state)
    proposal = state.get("pending_proposal") or {"tool_name": name, "arguments": args}

    decision = interrupt(proposal)  # pause ; à la reprise, renvoie reponse_humaine

    if decision.get("decision") == "validee":
        return {
            "pending_proposal": proposal,
            "human_decision": "validee",
            "human_proposed_action": None,
        }

    cp = decision.get("contre_proposition")
    # Contre-proposition mal formée (pas un objet, ou clés manquantes) -> refus simple.
    if cp is not None and (
        not isinstance(cp, dict) or "tool_name" not in cp or "arguments" not in cp
    ):
        cp = None
    if cp:
        try:
            validate_args(cp["tool_name"], cp["arguments"])
        except (ValidationError, KeyError, TypeError) as exc:
            return {
                "pending_proposal": None,
                "human_decision": None,
                "human_proposed_action": None,
                "messages": [
                    ToolMessage(
                        tool_call_id=call_id,
                        content=(
                            "Contre-proposition invalide : "
                            f"{exc}. Propose une action corrigée."
                        ),
                    ),
                    *_extra_tool_call_stubs(state),
                ],
            }
        return {
            "pending_proposal": None,
            "human_decision": "refusee",
            "human_proposed_action": cp,
        }

    motif = decision.get("motif_refus")
    return {
        "pending_proposal": None,
        "human_decision": "refusee",
        "motif_refus": motif,
        "messages": [
            ToolMessage(
                tool_call_id=call_id,
                content=(
                    f"Proposition refusée. Motif : {motif or 'non précisé'}. "
                    "Propose une alternative qui en tient compte."
                ),
            ),
            *_extra_tool_call_stubs(state),
        ],
    }


def route_after_gate(state: AgentState) -> str:
    if state.get("human_proposed_action"):
        return "act"
    if state.get("human_decision") == "validee" and state.get("pending_proposal"):
        return "act"
    return "think"


def act(state: AgentState) -> dict:
    tid = state["thread_id"]

    # tool_call_id TOUJOURS pris sur l'AIMessage-avec-tool_calls courant (R2),
    # y compris pour une contre-proposition humaine : celle-ci ne change QUE
    # l'outil/les args exécutés, jamais l'id renvoyé dans le ToolMessage.
    _, _, call_id = _pending_tool_call(state)

    hpa = state.get("human_proposed_action")
    if hpa:
        name, args = hpa["tool_name"], hpa["arguments"]
        origine = "humain_contre_proposition"
    elif state.get("pending_proposal"):
        name = state["pending_proposal"]["tool_name"]
        args = state["pending_proposal"]["arguments"]
        origine = "humain_validee"
    else:
        name, args, _ = _pending_tool_call(state)
        origine = "auto"

    try:
        new_df, env = dispatch(name, store.get_current(tid), args)
    except Exception as e:  # noqa: BLE001 — l'échec devient une enveloppe d'erreur uniforme
        new_df, env = None, {
            "status": "error",
            "summary": f"Échec inattendu de l'outil {name} : {e}",
            "metrics": {},
            "detail": None,
        }
    if new_df is not None and env["status"] == "ok":
        store.set_current(tid, new_df)

    upd: dict = {
        "messages": [
            ToolMessage(
                tool_call_id=call_id,
                content=json.dumps(
                    {
                        "status": env["status"],
                        "summary": env["summary"],
                        "metrics": env["metrics"],
                    },
                    ensure_ascii=False,
                ),
            ),
            *_extra_tool_call_stubs(state),
        ],
        "pending_proposal": None,
        "human_decision": None,
        "motif_refus": None,
        "human_proposed_action": None,
    }

    if env["status"] == "ok" and name in TOOL_TO_CHECKLIST:
        upd["checklist"] = {**state["checklist"], TOOL_TO_CHECKLIST[name]: True}

    if env["status"] == "ok":
        is_write = name in WRITE_TOOLS
        upd["actions_log"] = [
            {
                "outil": name,
                "parametres": {k: v for k, v in (args or {}).items() if k != "justification"},
                "justification": (args or {}).get("justification", ""),
                "metriques": env["metrics"],
                "timestamp": datetime.now().isoformat(),
                "type": "modification" if is_write else "lecture",
                "origine": origine if is_write else "auto",
            }
        ]

    return upd


def _finish(state: AgentState, capped: bool) -> dict:
    tid = state["thread_id"]
    try:
        env = generate_report(
            store.get_original(tid), store.get_current(tid), state["actions_log"]
        )
        md = env["detail"]["markdown"]
        summary = env["summary"]
    except Exception as e:  # noqa: BLE001 — le graphe doit terminer malgré un rapport en échec
        md = f"Échec de la génération du rapport : {e}"
        summary = md
    if capped:
        md = md.rstrip() + "\n\n## Note sur la génération\n\n" + _CAP_MESSAGE + "\n"
    return {
        "final_report": md,
        "task_done": True,
        "messages": [AIMessage(content=summary)],
    }


def report(state: AgentState) -> dict:
    return _finish(state, capped=False)


def force_report(state: AgentState) -> dict:
    return _finish(state, capped=True)


def nudge(state: AgentState) -> dict:
    """Tour LLM sans appel d'outil et plafond non atteint : on relance `think` avec
    un rappel explicite plutôt que de terminer le run à vide. Le garde-fou borne
    toujours la boucle (chaque `think` incrémente -> plafond -> `force_report`)."""
    return {"messages": [HumanMessage(content=_NUDGE_MESSAGE)]}


# ---------------------------------------------------------------- graph


def build_graph():
    """Graphe compilé, mémoïsé au niveau module (usage applicatif)."""
    global _GRAPH
    if _GRAPH is not None:
        return _GRAPH

    g = StateGraph(AgentState)
    g.add_node("think", think)
    g.add_node("act", act)
    g.add_node("human_gate", human_gate)
    g.add_node("force_report", force_report)
    g.add_node("report", report)
    g.add_node("nudge", nudge)

    g.add_edge(START, "think")
    g.add_conditional_edges(
        "think",
        route_after_think,
        {
            "act": "act",
            "human_gate": "human_gate",
            "report": "report",
            "force_report": "force_report",
            "nudge": "nudge",
        },
    )
    g.add_conditional_edges("human_gate", route_after_gate, {"act": "act", "think": "think"})
    g.add_edge("nudge", "think")
    g.add_edge("act", "think")
    g.add_edge("force_report", END)
    g.add_edge("report", END)

    _GRAPH = g.compile(checkpointer=InMemorySaver())
    return _GRAPH


def reset_graph() -> None:
    """R3 : remet `_GRAPH` à None ET vide `agent.store.DF_STORE`, pour que
    ni le checkpointer ni les dataframes ne fuient d'un test à l'autre."""
    global _GRAPH
    _GRAPH = None
    store.DF_STORE.clear()
    _PENDING_QA.clear()


# ---------------------------------------------------------------- app-facing helpers


def _graph_state(thread_id: str) -> dict:
    return build_graph().get_state(cfg(thread_id)).values


def _snapshot(thread_id: str):
    return build_graph().get_state(cfg(thread_id))


def start_run(thread_id: str, df) -> dict:
    store.init_thread(thread_id, df)
    state = initial_state(thread_id)
    # Seed : `think` doit toujours voir >= 1 message non-système, sinon le premier
    # appel Anthropic renvoie 400 ("at least one message required").
    state["messages"] = [
        HumanMessage(
            content=(
                f"Voici le jeu de données à préparer : {len(df)} lignes, "
                f"{len(df.columns)} colonnes ({', '.join(map(str, df.columns))}). "
                "Commence par le profilage (profile_dataset), puis suis l'ordre des opérations."
            )
        )
    ]
    build_graph().invoke(state, cfg(thread_id))
    return _graph_state(thread_id)


def resume_run(thread_id: str, reponse_humaine: dict) -> dict:
    """`reponse_humaine` : {"decision": "validee"|"refusee",
    "motif_refus": str|None, "contre_proposition": {"tool_name","arguments"}|None}."""
    build_graph().invoke(Command(resume=reponse_humaine), cfg(thread_id))
    return _graph_state(thread_id)


def submit_question(thread_id: str, question: str) -> str:
    """Question hors-bande — R1 : NE touche PAS au graphe. `graph.update_state`
    pendant une pause `interrupt()` purge la tâche `human_gate` en attente, si bien
    que le `resume_run` suivant ne reprend rien et le run se termine sans exécuter
    l'écriture (bouton Valider silencieusement inopérant). On répond ici en lecture
    seule sur le df courant, on empile la paire Q/R dans `_PENDING_QA` (drainée par
    `think` au prochain cycle) et on renvoie la réponse. La proposition en attente
    et l'interruption restent strictement intactes."""
    answer = _answer_question(store.get_current(thread_id), question)
    _PENDING_QA.setdefault(thread_id, []).append((question, answer))
    return answer


def get_pending_proposal(thread_id: str):
    """Proposition en attente {"tool_name": str, "arguments": dict} ou None.

    Lue depuis l'état (pré-stagée par `think` avant la pause). Repli sur la valeur
    de l'interruption du snapshot si jamais elle n'y était pas encore."""
    values = _graph_state(thread_id)
    prop = values.get("pending_proposal")
    if prop:
        return prop
    snap = _snapshot(thread_id)
    for itr in getattr(snap, "interrupts", ()) or ():
        val = getattr(itr, "value", None)
        if isinstance(val, dict) and "tool_name" in val:
            return val
    return None


def get_report(thread_id: str):
    return _graph_state(thread_id).get("final_report")

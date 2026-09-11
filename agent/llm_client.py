"""Point d'abstraction unique pour le LLM. Swap Claude → Ollama se fait ici et nulle part ailleurs."""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic

import config


def call_llm(messages: list, tools: list):
    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY manquante — renseigner le fichier .env (voir .env.example).")
    model = ChatAnthropic(
        model=config.MODEL_NAME,
        api_key=config.ANTHROPIC_API_KEY,
        temperature=0,
        max_tokens=4096,
    ).bind_tools(
        tools,
        # La boucle ReAct traite un seul outil par cycle (think -> un appel -> observation).
        # Sans ce flag, Anthropic peut renvoyer plusieurs tool_use dans un même tour ; seul
        # tool_calls[0] serait traité et les autres resteraient sans tool_result, ce que
        # l'API rejette au tour suivant ("tool_use ids were found without tool_result blocks").
        parallel_tool_calls=False,
    )
    try:
        return model.invoke(messages)
    except Exception:
        return model.invoke(messages)  # une tentative de retry avant de laisser remonter

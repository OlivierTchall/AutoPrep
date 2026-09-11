import pytest
from agent.prompts import SYSTEM_PROMPT


def test_system_prompt_contents():
    for token in ["AutoPrep Agent", "ORDRE DES OPÉRATIONS", "CHECKLIST DE SORTIE",
                  "RÈGLE DE VALIDATION", "QUESTIONS HORS-BANDE", "entièrement en français"]:
        assert token in SYSTEM_PROMPT


def test_call_llm_requires_key(monkeypatch):
    import agent.llm_client as lc
    monkeypatch.setattr(lc.config, "ANTHROPIC_API_KEY", None)
    with pytest.raises(RuntimeError):
        lc.call_llm([], [])


def test_call_llm_disables_parallel_tool_calls(monkeypatch):
    # Bug live constaté avec l'API réelle : sans ce flag, le modèle peut renvoyer
    # plusieurs tool_calls dans un même tour, ce que la boucle ReAct (un outil par
    # cycle) ne gère pas -> Anthropic rejette le tour suivant ("tool_use ids were
    # found without tool_result blocks"). On vérifie ici, sans appel réseau, que
    # `call_llm` demande bien un seul tool_call par tour.
    import agent.llm_client as lc

    monkeypatch.setattr(lc.config, "ANTHROPIC_API_KEY", "sk-ant-test")
    captured = {}

    class FakeBound:
        def invoke(self, messages):
            return "sentinel-response"

    class FakeChatAnthropic:
        def __init__(self, **kwargs):
            pass

        def bind_tools(self, tools, **kwargs):
            captured["tools"] = tools
            captured["kwargs"] = kwargs
            return FakeBound()

    monkeypatch.setattr(lc, "ChatAnthropic", FakeChatAnthropic)
    result = lc.call_llm(["msg"], ["tool"])
    assert result == "sentinel-response"
    assert captured["kwargs"].get("parallel_tool_calls") is False

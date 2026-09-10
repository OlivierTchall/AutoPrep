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

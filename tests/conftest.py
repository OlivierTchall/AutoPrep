import pytest
from langchain_core.messages import AIMessage
from data.generate_synthetic import generate_synthetic, write_csv


@pytest.fixture
def raw_df():
    return generate_synthetic(1000)


class ScriptedLLM:
    """Remplace call_llm : renvoie une séquence d'AIMessage préparés. Chaque entrée est
    soit un str (prose, pas de tool_call), soit (tool_class_name, args_dict)."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def __call__(self, messages, tools):
        item = self.script[self.calls] if self.calls < len(self.script) else "Fin."
        self.calls += 1
        if isinstance(item, str):
            return AIMessage(content=item)
        name, args = item
        return AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": f"call_{self.calls}"}],
        )


@pytest.fixture
def scripted(monkeypatch):
    def _install(script):
        llm = ScriptedLLM(script)
        import agent.loop as loop_mod

        monkeypatch.setattr(loop_mod, "call_llm", llm)
        return llm

    return _install


@pytest.fixture
def synthetic_csv(tmp_path, raw_df):
    p = tmp_path / "ventes_pme.csv"
    write_csv(raw_df, str(p))
    return str(p)

import pandas as pd
from agent.state import initial_state, CHECKLIST_KEYS, TOOL_TO_CHECKLIST
from agent import store


def test_initial_state_defaults():
    s = initial_state("t1")
    assert s["iteration_count"] == 0
    assert s["max_iterations"] == 15
    assert set(s["checklist"]) == set(CHECKLIST_KEYS)
    assert not any(s["checklist"].values())
    assert s["pending_proposal"] is None and s["actions_log"] == []


def test_store_roundtrip():
    df = pd.DataFrame({"a": [1, 2, 3]})
    store.init_thread("t2", df)
    store.set_current("t2", df.assign(b=4))
    assert list(store.get_current("t2").columns) == ["a", "b"]
    assert list(store.get_original("t2").columns) == ["a"]


def test_tool_to_checklist_covers_write_tools():
    from tools.registry import WRITE_TOOLS
    assert set(TOOL_TO_CHECKLIST) == WRITE_TOOLS | {"profile_dataset"}

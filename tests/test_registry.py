import pytest
from tools import registry


def test_write_tools_are_exactly_five():
    assert registry.WRITE_TOOLS == {
        "fix_column_types", "handle_duplicates", "handle_missing_values",
        "treat_outliers", "engineer_features",
    }


def test_registry_covers_all_ten_tools():
    assert set(registry.TOOL_REGISTRY) == registry.WRITE_TOOLS | registry.READ_TOOLS
    assert len(registry.TOOL_REGISTRY) == 10
    for name, meta in registry.TOOL_REGISTRY.items():
        assert meta["write"] == (name in registry.WRITE_TOOLS)
        assert meta["libelle"] and isinstance(meta["params_affiches"], list)
        assert "justification" not in meta["params_affiches"]


def test_llm_tools_list_has_ten_models():
    assert len(registry.LLM_TOOLS) == 10


def test_every_llm_tool_model_has_a_docstring():
    # I1 : bind_tools transmet `__doc__` comme description à Claude — aucune ne doit être vide.
    for model in registry.LLM_TOOLS:
        assert model.__doc__ and model.__doc__.strip()


def test_validate_args_rejects_bad_enum():
    with pytest.raises(Exception):
        registry.validate_args("handle_missing_values", {
            "colonnes": ["categorie"], "strategie": "telepathie", "justification": "x",
        })


def test_validate_args_requires_constant_value():
    with pytest.raises(Exception):
        registry.validate_args("handle_missing_values", {
            "colonnes": ["categorie"], "strategie": "constant", "justification": "x",
        })


def test_envelope_shape():
    e = registry.envelope("ok", "fait", {"avant": 10, "apres": 8})
    assert set(e) == {"status", "summary", "metrics", "detail"}

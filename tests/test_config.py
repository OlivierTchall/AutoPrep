"""Tests for config module."""


def test_config_constants():
    import config
    assert config.MAX_ITERATIONS == 15
    assert isinstance(config.MODEL_NAME, str) and config.MODEL_NAME
    assert config.OUTLIER_IQR_K == 1.5
    assert config.REPORT_PATH == "rapport.md"


def test_env_is_gitignored():
    from pathlib import Path
    assert ".env" in Path(".gitignore").read_text(encoding="utf-8").splitlines()

import pandas as pd
import pytest
from agent.upload import read_uploaded_csv, validate_dataframe


def test_reads_semicolon_utf8(raw_df, tmp_path):
    p = tmp_path / "a.csv"
    raw_df.to_csv(p, sep=";", index=False, encoding="utf-8")
    df = read_uploaded_csv(p.read_bytes())
    assert list(df.columns) == list(raw_df.columns)


def test_reads_latin1_fallback(tmp_path):
    p = tmp_path / "b.csv"
    pd.DataFrame({"région": ["Île-de-France"], "montant": [10]}).to_csv(p, sep=";", index=False, encoding="latin-1")
    df = read_uploaded_csv(p.read_bytes())
    assert "région" in df.columns


def test_validate_rejects_empty():
    with pytest.raises(ValueError):
        validate_dataframe(pd.DataFrame())

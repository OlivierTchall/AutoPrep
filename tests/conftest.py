import pytest
from data.generate_synthetic import generate_synthetic, write_csv


@pytest.fixture
def raw_df():
    return generate_synthetic(1000)


@pytest.fixture
def synthetic_csv(tmp_path, raw_df):
    p = tmp_path / "ventes_pme.csv"
    write_csv(raw_df, str(p))
    return str(p)

# AutoPrep Agent MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LangGraph ReAct agent that autonomously cleans an uploaded French-SME sales dataset, pausing for human validation before every data-modifying tool call, and produces a Markdown valorization report.

**Architecture:** A LangGraph `StateGraph` with `think` / `act` / `human_gate` nodes. `think` calls an LLM (Claude via `langchain-anthropic`) bound to typed tools; routing sends read-only tools straight to `act`, and the 5 write tools through `human_gate`, which uses LangGraph `interrupt()` to pause until a Streamlit button click resumes it. Dataframes live in a module-level dict keyed by `thread_id`, never in graph state or `messages`.

**Tech Stack:** Python 3.14, `uv`, LangGraph 1.x, `langchain-anthropic`, `pandas` 3.x, `pydantic` 2.x, `plotly`, `streamlit`, `faker`, `pytest`.

**Spec:** `autoprep-agent-architecture.md` (repo root) — the plan argues from this spec; executors read both.

---

## Global Constraints

- **Python floor:** `requires-python = ">=3.14"` (keep the existing pyproject value; the spec's "3.10+" is a lower bound only).
- **Dependency manager:** `uv` only. No `requirements.txt`. `uv sync` must reproduce the environment.
- **Secrets:** the Anthropic API key lives only in `.env` / environment (`ANTHROPIC_API_KEY`). `.env` MUST be git-ignored **before** any `.env` file is created. Only `.env.example` is committed.
- **LLM abstraction:** every LLM call goes through `call_llm()` in `agent/llm_client.py`. No other module imports `langchain_anthropic` or constructs a chat model.
- **HITL is non-negotiable:** the 5 tools in `WRITE_TOOLS` (`fix_column_types`, `handle_duplicates`, `handle_missing_values`, `treat_outliers`, `engineer_features`) MUST pause for human validation before executing. Read-only tools execute freely.
- **Dataframes never enter graph state or `messages`.** They live in `agent/store.py::DF_STORE[thread_id]`. Tool results sent to the LLM carry only `summary` + `metrics`.
- **Common tool return envelope:** `{"status": "ok"|"error", "summary": str, "metrics": dict, "detail": ...}`. Every write tool populates before/after values in `metrics`.
- **Iteration guardrail:** `max_iterations = 15`, hard-coded in `config.py`. `iteration_count` increments only on `think` node execution, never during the `human_gate` pause. On reaching the cap the agent emits `"Itération maximale atteinte (15) — je génère le rapport avec l'état actuel."` and routes to `generate_report` — a visible normal graph path, not a silent failure.
- **Mandated tool order is prompt-enforced only, never hard-coded:** profile → fix_column_types → handle_duplicates → handle_missing_values → detect/treat_outliers → engineer_features. Exit only via the explicit checklist.
- **Language:** the agent reasons, speaks, and writes the report entirely in French.
- **French file quirks:** upload validation auto-detects CSV separator (`;` vs `,`) and falls back `utf-8` → `latin-1`.
- **Synthetic generator reproducibility:** seed both `Faker.seed(42)` and `np.random.seed(42)`; write `date_commande` as raw `str` on CSV export.

---

## Deviations & open points to flag to Olivier (architect)

These are places where the spec is silent or the current scaffold conflicts with it. The plan picks a provisional resolution so it is executable; **Olivier reviews this section before execution.**

1. **Project layout — RESOLVED 2026-09-10 (Olivier):** follow the architecture doc's tree (`app.py`, `agent/`, `tools/`, `data/`, `tests/`, `config.py` at repo root). `pyproject.toml` is reworked to `[tool.uv] package = false` (Streamlit app, not a distributable package); `[build-system]` and `[project.scripts]` are removed; `src/autoprep/` is deleted. `CLAUDE.md`'s Commands section is updated to drop `uv run autoprep` (primary entry is `uv run streamlit run app.py`).
2. **LLM SDK — RESOLVED 2026-09-10 (Olivier):** `call_llm()` wraps `ChatAnthropic` from `langchain-anthropic` (already in pyproject), not the raw `anthropic` SDK. `pydantic` is added as an explicit dependency; `pytest` is added as a dev dependency.
3. **`agent/store.py` (new file, not in the spec tree):** the spec mandates "un dictionnaire au niveau module, indexé par `thread_id`" but names no file. This plan puts it in `agent/store.py`. *Flag for confirmation.*
4. **Initial snapshot storage:** the spec says `profile_dataset` "déclenche le snapshot initial" but the documented state schema has no field for it. This plan does **not** add a state field: `df_original` is preserved untouched in `DF_STORE`, and `generate_report` profiles both `df_original` and `df_current` itself. *Flag for confirmation.*
5. **`treat_outliers` and business-rule violations (negative `quantite`) — RESOLVED 2026-09-10 (Olivier):** the spec routes negative quantities to `treat_outliers` (step 5) but `treat_outliers`'s enum has no bound parameter, and IQR/z-score will not reliably flag a handful of negatives. Approved resolution: extend `treat_outliers` with one optional closed parameter `bornes_valides: {"min"?: float, "max"?: float} | None`; when given, rows outside the range are the target set, otherwise the IQR/z-score mask is used. Stays 5 write tools. Propagation: `tools/registry.py` pydantic model + `TOOL_REGISTRY["treat_outliers"]["params_affiches"]` (add `bornes_valides`), README known-limitations note. The architecture doc `autoprep-agent-architecture.md` should get this appended to its `treat_outliers` row on the next doc pass.
6. **`engineer_features` `formula` evaluation:** implemented via `pandas.DataFrame.eval` on an expression restricted (by regex validation) to column names, numbers, and arithmetic operators `+ - * / ( )`. No arbitrary Python. *Flag as interpretation.*
7. **Test-time LLM:** `tests/` monkeypatch `call_llm` with a scripted sequence of `AIMessage`s (no live API in CI). This is the intended use of the `call_llm()` seam.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Deps (add `pydantic`; dev group `pytest`), `[tool.uv] package = false`, `[tool.pytest.ini_options]` with `pythonpath = ["."]`. |
| `.gitignore` | Add `.env`. |
| `.env.example` | `ANTHROPIC_API_KEY=` placeholder. |
| `config.py` | `MAX_ITERATIONS = 15`, `MODEL_NAME`, `ANTHROPIC_API_KEY` (from env), outlier thresholds, `REPORT_PATH`. Non-secret behaviour constants + the one secret read from env. |
| `data/generate_synthetic.py` | Faker generator (seeded) → `data/ventes_pme.csv`. `main()` CLI. |
| `data/__init__.py` | Namespace marker. |
| `tools/__init__.py` | Namespace marker. |
| `tools/registry.py` | Envelope helper, pydantic arg schemas for all 10 tools, `TOOL_REGISTRY`, `WRITE_TOOLS`, `READ_TOOLS`, `LLM_TOOLS` (schema list for `bind_tools`), `dispatch(name, df, args)`. |
| `tools/profiling.py` | `profile_dataset(df)`. |
| `tools/typing_.py` | `fix_column_types(df, colonnes, types_cibles, justification)` + tolerant date parser. |
| `tools/cleaning.py` | `handle_duplicates`, `handle_missing_values`. |
| `tools/outliers.py` | `detect_outliers`, `treat_outliers`. |
| `tools/features.py` | `engineer_features` + safe formula evaluator. |
| `tools/query.py` | `query_dataframe` (4 parameterised operations). |
| `tools/charts.py` | `plot_chart` (plotly). |
| `tools/reporting.py` | `generate_report(df_original, df_current, actions_log)` → Markdown + writes `rapport.md`. |
| `agent/__init__.py` | Namespace marker. |
| `agent/store.py` | `DF_STORE: dict[str, dict]`, `init_thread`, `get_current`, `set_current`, `get_original`. |
| `agent/state.py` | `AgentState` TypedDict + `initial_state()`. |
| `agent/prompts.py` | `SYSTEM_PROMPT` (verbatim from spec). |
| `agent/llm_client.py` | `call_llm(messages, tools)`. |
| `agent/loop.py` | `build_graph()`, `think`, `act`, `human_gate` nodes, routers, `run_turn` / `resume_turn` helpers. |
| `app.py` | Streamlit UI: upload + encoding/separator detection, `thread_id`, pending-proposal panel with Valider/Refuser, chat, report download. |
| `tests/conftest.py` | Fixtures: `synthetic_csv`, `raw_df`, `ScriptedLLM`. |
| `tests/test_*.py` | One per tool module + `test_loop.py` + `test_scenario.py` (DoD). |

---

### Task 1: Project setup, config, secrets hygiene

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Create: `.env.example`
- Create: `config.py`
- Create: `agent/__init__.py`, `tools/__init__.py`, `data/__init__.py`
- Delete: `src/` (entire directory)
- Modify: `CLAUDE.md` (Commands section only)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `config.MAX_ITERATIONS: int = 15`, `config.MODEL_NAME: str`, `config.ANTHROPIC_API_KEY: str | None`, `config.OUTLIER_IQR_K: float = 1.5`, `config.OUTLIER_Z: float = 3.0`, `config.REPORT_PATH: str = "rapport.md"`, `config.SYNTHETIC_CSV: str = "data/ventes_pme.csv"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
def test_config_constants():
    import config
    assert config.MAX_ITERATIONS == 15
    assert isinstance(config.MODEL_NAME, str) and config.MODEL_NAME
    assert config.OUTLIER_IQR_K == 1.5
    assert config.REPORT_PATH == "rapport.md"


def test_env_is_gitignored():
    from pathlib import Path
    assert ".env" in Path(".gitignore").read_text(encoding="utf-8").splitlines()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: config` / `.env` not in `.gitignore`.

- [ ] **Step 3: Rework `pyproject.toml`**

Replace the `[build-system]` and `[project.scripts]` sections and extend deps:

```toml
[project]
name = "autoprep"
version = "0.1.0"
description = "AutoPrep Agent — HITL LangGraph agent for cleaning PME sales datasets"
readme = "README.md"
authors = [{ name = "TCHALLY", email = "tchallyolivier@gmail.com" }]
requires-python = ">=3.14"
dependencies = [
    "faker>=40.38.0",
    "langchain>=1.4.0",
    "langchain-anthropic>=1.7.1",
    "langgraph>=1.2.11",
    "numpy>=2.5.3",
    "pandas>=3.0.5",
    "plotly>=7.0.0",
    "pydantic>=2.9.0",
    "python-dotenv>=1.2.3",
    "streamlit>=1.63.0",
]

[dependency-groups]
dev = ["pytest>=8.3.0"]

[tool.uv]
package = false

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

(Remove `[build-system]` and `[project.scripts]` entirely.)

- [ ] **Step 4: Delete `src/`, add package markers, write `config.py`, `.env.example`, fix `.gitignore`**

```bash
git rm -r src
```

`agent/__init__.py`, `tools/__init__.py`, `data/__init__.py`: each a single line — `"""AutoPrep package."""`

`.env.example`:
```
# Compte Claude Console (console.anthropic.com) — facturé à l'usage, distinct de l'abonnement Claude Pro.
ANTHROPIC_API_KEY=
```

Append to `.gitignore`:
```
# Secrets
.env
```

`config.py`:
```python
"""Constantes de comportement (non secrètes) + lecture de la clé API depuis l'environnement."""
import os
from dotenv import load_dotenv

load_dotenv()

MAX_ITERATIONS = 15
MODEL_NAME = os.getenv("AUTOPREP_MODEL", "claude-haiku-4-5-20251001")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

OUTLIER_IQR_K = 1.5
OUTLIER_Z = 3.0

SYNTHETIC_CSV = "data/ventes_pme.csv"
REPORT_PATH = "rapport.md"
```

- [ ] **Step 5: Update `CLAUDE.md` Commands section**

Replace the `uv run autoprep` line with:
```
uv run streamlit run app.py      # run the Streamlit UI (primary entry point)
uv run python -m data.generate_synthetic   # regenerate the synthetic dataset
uv run pytest                     # run the test suite
```

- [ ] **Step 6: Run tests + sync**

Run: `uv sync && uv run pytest tests/test_config.py -v`
Expected: PASS. `uv sync` resolves without error.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .gitignore .env.example config.py agent tools data CLAUDE.md tests/test_config.py
git commit -m "chore: rework project layout, config, secrets hygiene"
```

---

### Task 2: Synthetic PME dataset generator

**Files:**
- Create: `data/generate_synthetic.py`
- Test: `tests/test_generate_synthetic.py`
- Test: `tests/conftest.py` (add `synthetic_csv` + `raw_df` fixtures)

**Interfaces:**
- Produces: `generate_synthetic(n: int = 1000) -> pandas.DataFrame` — columns `order_id, date_commande, client, produit, categorie, quantite, prix_unitaire, montant_total, region, mode_paiement`; `date_commande` is `object` (str) dtype.
- Produces: `write_csv(df: pandas.DataFrame, path: str) -> None` — `sep=";"`, `encoding="utf-8"`, `date_commande` written verbatim as string.
- Produces: `main() -> None` — writes `config.SYNTHETIC_CSV`.
- Consumes (tests): nothing from earlier tasks except `config`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_generate_synthetic.py
import pandas as pd
from data.generate_synthetic import generate_synthetic


def test_reproducible():
    a = generate_synthetic(1000)
    b = generate_synthetic(1000)
    pd.testing.assert_frame_equal(a, b)


def test_shape_and_columns():
    df = generate_synthetic(1000)
    assert len(df) == 1000
    assert list(df.columns) == [
        "order_id", "date_commande", "client", "produit", "categorie",
        "quantite", "prix_unitaire", "montant_total", "region", "mode_paiement",
    ]


def test_injected_anomalies():
    df = generate_synthetic(1000)
    # dates kept as raw strings, two formats mixed
    assert df["date_commande"].dtype == object
    slash = df["date_commande"].str.contains("/").mean()
    assert 0.10 < slash < 0.25
    # missing values ~6% on categorie and mode_paiement
    assert 0.03 < df["categorie"].isna().mean() < 0.09
    assert 0.03 < df["mode_paiement"].isna().mean() < 0.09
    # ~2% duplicated rows
    assert 0.01 <= df.duplicated().mean() <= 0.04
    # business-rule violation: some negative quantities
    assert (df["quantite"] < 0).sum() >= 3
    # montant_total inconsistent with quantite*prix_unitaire on some rows
    prod = (df["quantite"] * df["prix_unitaire"]).round(2)
    assert (prod != df["montant_total"].round(2)).sum() >= 20
```

Add to `tests/conftest.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_generate_synthetic.py -v`
Expected: FAIL — `ModuleNotFoundError: data.generate_synthetic`.

- [ ] **Step 3: Implement `data/generate_synthetic.py`**

```python
"""Générateur de jeu de données synthétique 'Ventes PME'. Seed fixe pour reproductibilité."""
from __future__ import annotations

import numpy as np
import pandas as pd
from faker import Faker

import config

SEED = 42
CATEGORIES = ["Informatique", "Bureau", "Mobilier", "Consommables", "Réseau"]
PRODUITS = {
    "Informatique": ["Ordinateur portable", "Écran 27\"", "Clavier mécanique", "Souris sans fil"],
    "Bureau": ["Ramette papier A4", "Cartouche encre", "Agrafeuse", "Bloc-notes"],
    "Mobilier": ["Chaise ergonomique", "Bureau assis-debout", "Caisson tiroirs"],
    "Consommables": ["Café 1kg", "Gobelets x100", "Lingettes écran"],
    "Réseau": ["Switch 8 ports", "Câble RJ45 5m", "Point d'accès WiFi"],
}
REGIONS = ["Île-de-France", "Auvergne-Rhône-Alpes", "Occitanie", "Bretagne", "Grand Est"]
PAIEMENTS = ["Virement", "Carte", "Chèque", "Prélèvement"]


def generate_synthetic(n: int = 1000) -> pd.DataFrame:
    fake = Faker("fr_FR")
    Faker.seed(SEED)
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    clients = [fake.company() for _ in range(max(20, n // 25))]

    rows = []
    for i in range(n):
        cat = CATEGORIES[rng.integers(len(CATEGORIES))]
        prod = PRODUITS[cat][rng.integers(len(PRODUITS[cat]))]
        qte = int(rng.integers(1, 40))
        pu = round(float(rng.uniform(2, 900)), 2)
        # date: 85% ISO AAAA-MM-JJ, 15% JJ/MM/AAAA
        d = fake.date_between(start_date="-2y", end_date="today")
        if rng.random() < 0.15:
            date_str = d.strftime("%d/%m/%Y")
        else:
            date_str = d.strftime("%Y-%m-%d")
        rows.append({
            "order_id": 100000 + i,
            "date_commande": date_str,
            "client": clients[rng.integers(len(clients))],
            "produit": prod,
            "categorie": cat,
            "quantite": qte,
            "prix_unitaire": pu,
            "montant_total": round(qte * pu, 2),
            "region": REGIONS[rng.integers(len(REGIONS))],
            "mode_paiement": PAIEMENTS[rng.integers(len(PAIEMENTS))],
        })

    df = pd.DataFrame(rows)

    # --- anomalies injectées (indices tirés du rng seedé) ---
    # 1. valeurs manquantes ~6% sur categorie et mode_paiement
    for col in ("categorie", "mode_paiement"):
        idx = rng.choice(n, size=int(0.06 * n), replace=False)
        df.loc[idx, col] = np.nan

    # 2. quantités négatives / aberrantes (violation règle métier) — ~1%
    neg_idx = rng.choice(n, size=max(5, int(0.01 * n)), replace=False)
    df.loc[neg_idx, "quantite"] = -rng.integers(1, 8, size=len(neg_idx))

    # 3. montant_total incohérent sur ~4% des lignes
    bad_idx = rng.choice(n, size=int(0.04 * n), replace=False)
    df.loc[bad_idx, "montant_total"] = (df.loc[bad_idx, "montant_total"] * rng.uniform(1.2, 2.0, size=len(bad_idx))).round(2)

    # 4. doublons ~2% (concaténer des copies de lignes existantes)
    dup_idx = rng.choice(n, size=int(0.02 * n), replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)
    df = df.iloc[:n].reset_index(drop=True)  # garder n lignes, doublons inclus

    return df


def write_csv(df: pd.DataFrame, path: str) -> None:
    out = df.copy()
    out["date_commande"] = out["date_commande"].astype(str)
    out.to_csv(path, sep=";", index=False, encoding="utf-8")


def main() -> None:
    df = generate_synthetic(1000)
    write_csv(df, config.SYNTHETIC_CSV)
    print(f"Écrit {len(df)} lignes dans {config.SYNTHETIC_CSV}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_generate_synthetic.py -v`
Expected: PASS. If `test_injected_anomalies` bounds fail, adjust injection rates (not the assertion bounds) until all pass.

- [ ] **Step 5: Generate the committed dataset**

Run: `uv run python -m data.generate_synthetic`
Expected: `data/ventes_pme.csv` created.

- [ ] **Step 6: Commit**

```bash
git add data/generate_synthetic.py data/ventes_pme.csv tests/test_generate_synthetic.py tests/conftest.py
git commit -m "feat: seeded synthetic PME dataset generator"
```

---

### Task 3: Tool schemas, registry, dispatcher

**Files:**
- Create: `tools/registry.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Produces:
  - `envelope(status, summary, metrics=None, detail=None) -> dict` — builds the common return envelope.
  - Pydantic models (all `pydantic.BaseModel`, all fields typed, French field docs): `ProfileDatasetArgs`, `DetectOutliersArgs`, `GenerateReportArgs`, `PlotChartArgs`, `QueryDataframeArgs`, `FixColumnTypesArgs`, `HandleDuplicatesArgs`, `HandleMissingValuesArgs`, `TreatOutliersArgs`, `EngineerFeaturesArgs`.
  - `WRITE_TOOLS: set[str] = {"fix_column_types", "handle_duplicates", "handle_missing_values", "treat_outliers", "engineer_features"}`
  - `READ_TOOLS: set[str] = {"profile_dataset", "detect_outliers", "generate_report", "plot_chart", "query_dataframe"}`
  - `TOOL_REGISTRY: dict[str, dict]` — per tool: `{"libelle": str, "params_affiches": list[str], "args_model": type[BaseModel], "write": bool}`.
  - `LLM_TOOLS: list[type[BaseModel]]` — the 10 arg models, for `ChatAnthropic.bind_tools`.
  - `validate_args(tool_name: str, args: dict) -> dict` — runs the pydantic model, returns `model.model_dump()`, raises `pydantic.ValidationError` on bad input.
  - `dispatch(tool_name: str, df: pandas.DataFrame, args: dict) -> tuple[pandas.DataFrame | None, dict]` — returns `(new_df_or_None, envelope)`. Read tools return `(None, envelope)`; write tools return `(new_df, envelope)`. Implemented as a lazy import table so this module has no import cycle with tool modules.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_registry.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: tools.registry`.

- [ ] **Step 3: Implement `tools/registry.py`**

```python
"""Schémas d'arguments (pydantic), classification lecture/écriture, TOOL_REGISTRY, dispatcher."""
from __future__ import annotations

import importlib
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


def envelope(status: str, summary: str, metrics: dict | None = None, detail: Any = None) -> dict:
    return {"status": status, "summary": summary, "metrics": metrics or {}, "detail": detail}


# ---------- arg models: read tools ----------
class ProfileDatasetArgs(BaseModel):
    pass


class DetectOutliersArgs(BaseModel):
    colonnes: list[str] = Field(..., description="Colonnes numériques à analyser.")
    methode: Literal["iqr", "zscore"] = Field("iqr", description="Méthode de détection.")


class GenerateReportArgs(BaseModel):
    pass


class PlotChartArgs(BaseModel):
    type: Literal["bar", "line", "histogram", "scatter", "box"]
    x: str
    y: str | None = None
    titre: str = ""


class _Periode(BaseModel):
    colonne_date: str
    mois: int | None = None
    annee: int | None = None


class QueryDataframeArgs(BaseModel):
    operation: Literal["groupby_agg", "filter_count", "describe_column", "top_n"]
    parametres: dict = Field(default_factory=dict)


# ---------- arg models: write tools ----------
class FixColumnTypesArgs(BaseModel):
    colonnes: list[str]
    types_cibles: dict[str, Literal["entier", "decimal", "date", "texte", "categorie"]]
    justification: str


class HandleDuplicatesArgs(BaseModel):
    sous_ensemble_colonnes: list[str] | None = None
    justification: str


class HandleMissingValuesArgs(BaseModel):
    colonnes: list[str]
    strategie: Literal["drop_rows", "mean", "median", "mode", "constant"]
    valeur_constante: Any | None = None
    justification: str

    @model_validator(mode="after")
    def _constant_needs_value(self):
        if self.strategie == "constant" and self.valeur_constante is None:
            raise ValueError("valeur_constante est requise quand strategie == 'constant'")
        return self


class _BornesValides(BaseModel):
    min: float | None = None
    max: float | None = None


class TreatOutliersArgs(BaseModel):
    colonnes: list[str]
    action: Literal["cap", "remove_rows", "set_nan"]
    justification: str
    methode: Literal["iqr", "zscore"] = "iqr"
    bornes_valides: _BornesValides | None = Field(
        None, description="Plage métier valide ; si fournie, les lignes hors plage sont la cible (violation de règle métier).",
    )


class _FeatureSpec(BaseModel):
    type: Literal["date_components", "ratio", "binned", "flag", "formula"]
    source: list[str]
    output: str
    params: dict = Field(default_factory=dict)


class EngineerFeaturesArgs(BaseModel):
    specification: _FeatureSpec
    justification: str


# ---------- registry ----------
_META = {
    "profile_dataset": ("Profiler le jeu de données", [], ProfileDatasetArgs, False),
    "detect_outliers": ("Détecter les valeurs extrêmes", ["colonnes", "methode"], DetectOutliersArgs, False),
    "generate_report": ("Générer le rapport final", [], GenerateReportArgs, False),
    "plot_chart": ("Tracer un graphique", ["type", "x", "y"], PlotChartArgs, False),
    "query_dataframe": ("Interroger les données", ["operation", "parametres"], QueryDataframeArgs, False),
    "fix_column_types": ("Corriger le type d'une colonne", ["colonnes", "types_cibles"], FixColumnTypesArgs, True),
    "handle_duplicates": ("Supprimer les doublons", ["sous_ensemble_colonnes"], HandleDuplicatesArgs, True),
    "handle_missing_values": ("Traiter les valeurs manquantes", ["colonnes", "strategie", "valeur_constante"], HandleMissingValuesArgs, True),
    "treat_outliers": ("Traiter les valeurs extrêmes", ["colonnes", "action", "bornes_valides"], TreatOutliersArgs, True),
    "engineer_features": ("Créer une variable dérivée", ["specification"], EngineerFeaturesArgs, True),
}

TOOL_REGISTRY = {
    name: {"libelle": lib, "params_affiches": params, "args_model": model, "write": write}
    for name, (lib, params, model, write) in _META.items()
}
WRITE_TOOLS = {n for n, m in TOOL_REGISTRY.items() if m["write"]}
READ_TOOLS = {n for n, m in TOOL_REGISTRY.items() if not m["write"]}
LLM_TOOLS = [m["args_model"] for m in TOOL_REGISTRY.values()]

# pydantic model class name -> tool name, so ChatAnthropic tool_calls (which use the
# model's class name) map back to our registry keys.
MODEL_NAME_TO_TOOL = {m["args_model"].__name__: n for n, m in TOOL_REGISTRY.items()}


def validate_args(tool_name: str, args: dict) -> dict:
    model = TOOL_REGISTRY[tool_name]["args_model"]
    return model.model_validate(args or {}).model_dump()


_DISPATCH_TABLE = {
    "profile_dataset": ("tools.profiling", "profile_dataset"),
    "detect_outliers": ("tools.outliers", "detect_outliers"),
    "plot_chart": ("tools.charts", "plot_chart"),
    "query_dataframe": ("tools.query", "query_dataframe"),
    "fix_column_types": ("tools.typing_", "fix_column_types"),
    "handle_duplicates": ("tools.cleaning", "handle_duplicates"),
    "handle_missing_values": ("tools.cleaning", "handle_missing_values"),
    "treat_outliers": ("tools.outliers", "treat_outliers"),
    "engineer_features": ("tools.features", "engineer_features"),
}


def dispatch(tool_name: str, df, args: dict):
    """Execute a tool. Returns (new_df_or_None, envelope). generate_report is handled by the graph, not here."""
    clean = validate_args(tool_name, args)
    mod_name, fn_name = _DISPATCH_TABLE[tool_name]
    fn = getattr(importlib.import_module(mod_name), fn_name)
    result = fn(df, **clean)
    if tool_name in WRITE_TOOLS:
        new_df, env = result
        return new_df, env
    return None, result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_registry.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/registry.py tests/test_registry.py
git commit -m "feat: tool arg schemas, registry, dispatcher"
```

---

### Task 4: `profile_dataset`

**Files:**
- Create: `tools/profiling.py`
- Test: `tests/test_profiling.py`

**Interfaces:**
- Consumes: `tools.registry.envelope`.
- Produces: `profile_dataset(df: pandas.DataFrame) -> dict` (read envelope). `detail` contains: `{"n_lignes": int, "n_colonnes": int, "types": {col: str}, "valeurs_manquantes": {col: {"n": int, "taux": float}}, "n_doublons": int, "numeriques": {col: {"min": float, "max": float, "mean": float, "std": float, "median": float}}, "apercu": [ {col: val} x5 ]}`. `summary` is a French sentence naming row count, dup count, and any numeric column whose `min < 0`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_profiling.py
from tools.profiling import profile_dataset


def test_profile_flags_synthetic_anomalies(raw_df):
    out = profile_dataset(raw_df)
    assert out["status"] == "ok"
    d = out["detail"]
    assert d["n_lignes"] == len(raw_df)
    assert d["n_doublons"] >= 1
    assert d["valeurs_manquantes"]["categorie"]["n"] > 0
    # min/max exposed per numeric column, and negative quantite visible
    assert d["numeriques"]["quantite"]["min"] < 0
    assert "quantite" in out["summary"].lower() or "négative" in out["summary"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_profiling.py -v`
Expected: FAIL — `ModuleNotFoundError: tools.profiling`.

- [ ] **Step 3: Implement `tools/profiling.py`**

```python
"""profile_dataset — lecture seule. Types, manquants, doublons, stats, min/max par colonne numérique."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tools.registry import envelope


def profile_dataset(df: pd.DataFrame) -> dict:
    n = len(df)
    manquants = {
        c: {"n": int(df[c].isna().sum()), "taux": round(float(df[c].isna().mean()), 4)}
        for c in df.columns
    }
    num_cols = df.select_dtypes(include=[np.number]).columns
    numeriques = {
        c: {
            "min": float(df[c].min()),
            "max": float(df[c].max()),
            "mean": round(float(df[c].mean()), 4),
            "std": round(float(df[c].std()), 4),
            "median": float(df[c].median()),
        }
        for c in num_cols
    }
    n_dup = int(df.duplicated().sum())
    negs = [c for c, s in numeriques.items() if s["min"] < 0]
    phrase = f"{n} lignes, {len(df.columns)} colonnes, {n_dup} doublon(s)."
    if negs:
        phrase += f" Valeurs négatives suspectes dans : {', '.join(negs)} (règle métier)."
    return envelope(
        "ok",
        phrase,
        metrics={"n_lignes": n, "n_doublons": n_dup},
        detail={
            "n_lignes": n,
            "n_colonnes": len(df.columns),
            "types": {c: str(t) for c, t in df.dtypes.items()},
            "valeurs_manquantes": manquants,
            "n_doublons": n_dup,
            "numeriques": numeriques,
            "apercu": df.head(5).to_dict(orient="records"),
        },
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_profiling.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/profiling.py tests/test_profiling.py
git commit -m "feat: profile_dataset tool"
```

---

### Task 5: `fix_column_types` with tolerant date parsing

**Files:**
- Create: `tools/typing_.py`
- Test: `tests/test_typing.py`

**Interfaces:**
- Consumes: `tools.registry.envelope`.
- Produces: `fix_column_types(df, colonnes: list[str], types_cibles: dict[str, str], justification: str) -> tuple[pandas.DataFrame, dict]`. Envelope `metrics` carries `{"avant": {col: dtype}, "apres": {col: dtype}, "non_convertis": {col: int}}`. Date parsing tries, in order: `%Y-%m-%d`, `%d/%m/%Y`, `%d-%m-%Y`, `%m/%d/%Y`, then `pd.to_datetime(errors="coerce", dayfirst=True)`; unparseable → `NaT`.
- Produces: `parse_dates_tolerant(s: pandas.Series) -> pandas.Series` (exported for reuse/testing).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_typing.py
import pandas as pd
from tools.typing_ import fix_column_types, parse_dates_tolerant


def test_parse_mixed_date_formats():
    s = pd.Series(["2024-03-15", "01/04/2024", "31/12/2023", "pas une date"])
    out = parse_dates_tolerant(s)
    assert out.dtype == "datetime64[ns]"
    assert out.isna().sum() == 1
    assert out.iloc[1] == pd.Timestamp("2024-04-01")


def test_fix_column_types_on_synthetic(raw_df):
    new, env = fix_column_types(
        raw_df, colonnes=["date_commande", "quantite"],
        types_cibles={"date_commande": "date", "quantite": "entier"},
        justification="Le profilage montre date_commande en texte et deux formats mélangés.",
    )
    assert env["status"] == "ok"
    assert new["date_commande"].dtype == "datetime64[ns]"
    assert str(new["quantite"].dtype).startswith("int") or str(new["quantite"].dtype).startswith("Int")
    assert env["metrics"]["apres"]["date_commande"] == "datetime64[ns]"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_typing.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `tools/typing_.py`**

```python
"""fix_column_types — outil de modification. Parsing tolérant multi-format pour les dates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tools.registry import envelope

_DATE_PATTERNS = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%m/%d/%Y"]


def parse_dates_tolerant(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    remaining = s.notna()
    for fmt in _DATE_PATTERNS:
        if not remaining.any():
            break
        parsed = pd.to_datetime(s[remaining], format=fmt, errors="coerce")
        got = parsed.notna()
        out.loc[parsed.index[got]] = parsed[got]
        remaining.loc[parsed.index[got]] = False
    if remaining.any():
        parsed = pd.to_datetime(s[remaining], errors="coerce", dayfirst=True)
        out.loc[parsed.index] = parsed
    return out


def _coerce(series: pd.Series, target: str) -> pd.Series:
    if target == "date":
        return parse_dates_tolerant(series)
    if target == "entier":
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    if target == "decimal":
        return pd.to_numeric(series, errors="coerce").astype("float64")
    if target == "texte":
        return series.astype("string")
    if target == "categorie":
        return series.astype("category")
    raise ValueError(f"type cible inconnu: {target}")


def fix_column_types(df, colonnes, types_cibles, justification):
    new = df.copy()
    avant, apres, non_convertis = {}, {}, {}
    for col in colonnes:
        if col not in new.columns:
            return df, envelope("error", f"Colonne inexistante : {col}.", detail={"colonne": col})
        target = types_cibles.get(col)
        if target is None:
            return df, envelope("error", f"Aucun type cible fourni pour {col}.")
        avant[col] = str(df[col].dtype)
        before_na = int(new[col].isna().sum())
        new[col] = _coerce(new[col], target)
        after_na = int(new[col].isna().sum())
        apres[col] = str(new[col].dtype)
        non_convertis[col] = max(0, after_na - before_na)
    total_bad = sum(non_convertis.values())
    return new, envelope(
        "ok",
        f"Types corrigés pour {', '.join(colonnes)}. {total_bad} valeur(s) non convertible(s) → manquantes.",
        metrics={"avant": avant, "apres": apres, "non_convertis": non_convertis},
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_typing.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/typing_.py tests/test_typing.py
git commit -m "feat: fix_column_types with tolerant date parsing"
```

---

### Task 6: `handle_duplicates` + `handle_missing_values`

**Files:**
- Create: `tools/cleaning.py`
- Test: `tests/test_cleaning.py`

**Interfaces:**
- Consumes: `tools.registry.envelope`.
- Produces: `handle_duplicates(df, sous_ensemble_colonnes: list[str] | None, justification: str) -> tuple[df, envelope]`. `metrics`: `{"avant": n, "apres": n, "supprimees": n}`.
- Produces: `handle_missing_values(df, colonnes: list[str], strategie: str, valeur_constante, justification: str) -> tuple[df, envelope]`. `metrics`: `{"avant": {col: n_na}, "apres": {col: n_na}, "lignes_avant": n, "lignes_apres": n}`. Strategies: `drop_rows` (drop rows with NA in any of `colonnes`), `mean`/`median` (numeric only — error envelope if a target col is non-numeric), `mode`, `constant` (fill with `valeur_constante`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cleaning.py
import numpy as np
import pandas as pd
from tools.cleaning import handle_duplicates, handle_missing_values


def test_handle_duplicates_removes_all(raw_df):
    new, env = handle_duplicates(raw_df, sous_ensemble_colonnes=None, justification="2% de doublons au profilage.")
    assert env["status"] == "ok"
    assert new.duplicated().sum() == 0
    assert env["metrics"]["supprimees"] == env["metrics"]["avant"] - env["metrics"]["apres"]


def test_missing_mode_fills_categorie(raw_df):
    new, env = handle_missing_values(raw_df, colonnes=["categorie", "mode_paiement"], strategie="mode",
                                     valeur_constante=None, justification="6% de manquants.")
    assert env["status"] == "ok"
    assert new["categorie"].isna().sum() == 0
    assert new["mode_paiement"].isna().sum() == 0


def test_missing_constant():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    new, env = handle_missing_values(df, colonnes=["a"], strategie="constant",
                                     valeur_constante=0, justification="x")
    assert new["a"].tolist() == [1.0, 0.0, 3.0]


def test_missing_mean_on_text_is_error(raw_df):
    new, env = handle_missing_values(raw_df, colonnes=["categorie"], strategie="mean",
                                     valeur_constante=None, justification="x")
    assert env["status"] == "error"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cleaning.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `tools/cleaning.py`**

```python
"""handle_duplicates + handle_missing_values — outils de modification."""
from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype

from tools.registry import envelope


def handle_duplicates(df, sous_ensemble_colonnes, justification):
    subset = sous_ensemble_colonnes or None
    if subset:
        missing = [c for c in subset if c not in df.columns]
        if missing:
            return df, envelope("error", f"Colonnes inexistantes : {missing}.")
    before = len(df)
    new = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    after = len(new)
    return new, envelope(
        "ok",
        f"{before - after} doublon(s) supprimé(s) ({before} → {after} lignes).",
        metrics={"avant": before, "apres": after, "supprimees": before - after},
    )


def handle_missing_values(df, colonnes, strategie, valeur_constante, justification):
    missing = [c for c in colonnes if c not in df.columns]
    if missing:
        return df, envelope("error", f"Colonnes inexistantes : {missing}.")
    avant = {c: int(df[c].isna().sum()) for c in colonnes}
    new = df.copy()
    lignes_avant = len(new)

    if strategie == "drop_rows":
        new = new.dropna(subset=colonnes).reset_index(drop=True)
    elif strategie in ("mean", "median"):
        for c in colonnes:
            if not is_numeric_dtype(new[c]):
                return df, envelope("error", f"Stratégie '{strategie}' impossible sur la colonne non numérique {c}.")
            val = new[c].mean() if strategie == "mean" else new[c].median()
            new[c] = new[c].fillna(val)
    elif strategie == "mode":
        for c in colonnes:
            m = new[c].mode(dropna=True)
            if not m.empty:
                new[c] = new[c].fillna(m.iloc[0])
    elif strategie == "constant":
        for c in colonnes:
            new[c] = new[c].fillna(valeur_constante)

    apres = {c: int(new[c].isna().sum()) for c in colonnes}
    return new, envelope(
        "ok",
        f"Valeurs manquantes traitées ({strategie}) sur {', '.join(colonnes)}.",
        metrics={"avant": avant, "apres": apres, "lignes_avant": lignes_avant, "lignes_apres": len(new)},
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cleaning.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/cleaning.py tests/test_cleaning.py
git commit -m "feat: handle_duplicates and handle_missing_values tools"
```

---

### Task 7: `detect_outliers` + `treat_outliers`

**Files:**
- Create: `tools/outliers.py`
- Test: `tests/test_outliers.py`

**Interfaces:**
- Consumes: `tools.registry.envelope`, `config.OUTLIER_IQR_K`, `config.OUTLIER_Z`.
- Produces: `detect_outliers(df, colonnes: list[str], methode: str = "iqr") -> dict` (read envelope). `detail`: `{col: {"n_outliers": int, "bornes": [low, high], "exemples": [..≤5..]}}`.
- Produces: `treat_outliers(df, colonnes: list[str], action: str, justification: str, methode: str = "iqr", bornes_valides: dict | None = None) -> tuple[df, envelope]`. Target mask: if `bornes_valides` given, rows where value `< min` or `> max`; else the IQR/z-score mask. Actions: `cap` (clip to fence or to `bornes_valides`), `remove_rows`, `set_nan`. `metrics`: `{"avant": {col: {"min","max"}}, "apres": {col: {"min","max"}}, "lignes_avant": n, "lignes_apres": n, "cibles": {col: n}}`.
- **NOTE:** `bornes_valides` is a provisional spec extension — see "Deviations" point 5.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_outliers.py
import numpy as np
import pandas as pd
from tools.outliers import detect_outliers, treat_outliers


def test_detect_outliers_iqr():
    df = pd.DataFrame({"x": [10, 11, 12, 10, 11, 9, 10, 500]})
    out = detect_outliers(df, colonnes=["x"], methode="iqr")
    assert out["status"] == "ok"
    assert out["detail"]["x"]["n_outliers"] == 1


def test_treat_outliers_business_rule_remove(raw_df):
    new, env = treat_outliers(raw_df, colonnes=["quantite"], action="remove_rows",
                              justification="Quantités négatives = violation de règle métier (profilage: min < 0).",
                              bornes_valides={"min": 0, "max": None})
    assert env["status"] == "ok"
    assert new["quantite"].min() >= 0
    assert env["metrics"]["lignes_apres"] < env["metrics"]["lignes_avant"]


def test_treat_outliers_cap_with_bounds():
    df = pd.DataFrame({"q": [-3, -1, 2, 5, 8]})
    new, env = treat_outliers(df, colonnes=["q"], action="cap", justification="x",
                              bornes_valides={"min": 0, "max": None})
    assert new["q"].min() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_outliers.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `tools/outliers.py`**

```python
"""detect_outliers (lecture) + treat_outliers (modification)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

import config
from tools.registry import envelope


def _fences(s: pd.Series, methode: str):
    s = s.dropna()
    if methode == "zscore":
        m, sd = s.mean(), s.std()
        return m - config.OUTLIER_Z * sd, m + config.OUTLIER_Z * sd
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - config.OUTLIER_IQR_K * iqr, q3 + config.OUTLIER_IQR_K * iqr


def detect_outliers(df, colonnes, methode="iqr"):
    detail = {}
    for c in colonnes:
        if c not in df.columns or not is_numeric_dtype(df[c]):
            return envelope("error", f"Colonne absente ou non numérique : {c}.")
        low, high = _fences(df[c], methode)
        mask = (df[c] < low) | (df[c] > high)
        detail[c] = {
            "n_outliers": int(mask.sum()),
            "bornes": [round(float(low), 4), round(float(high), 4)],
            "exemples": df.loc[mask, c].head(5).tolist(),
        }
    total = sum(v["n_outliers"] for v in detail.values())
    return envelope("ok", f"{total} valeur(s) extrême(s) détectée(s) ({methode}).", metrics={"total": total}, detail=detail)


def _target_mask(s: pd.Series, methode: str, bornes: dict | None):
    if bornes:
        lo = bornes.get("min")
        hi = bornes.get("max")
        m = pd.Series(False, index=s.index)
        if lo is not None:
            m |= s < lo
        if hi is not None:
            m |= s > hi
        return m, lo, hi
    low, high = _fences(s, methode)
    return (s < low) | (s > high), low, high


def treat_outliers(df, colonnes, action, justification, methode="iqr", bornes_valides=None):
    for c in colonnes:
        if c not in df.columns or not is_numeric_dtype(df[c]):
            return df, envelope("error", f"Colonne absente ou non numérique : {c}.")
    new = df.copy()
    lignes_avant = len(new)
    avant = {c: {"min": float(df[c].min()), "max": float(df[c].max())} for c in colonnes}
    cibles = {}
    drop_mask = pd.Series(False, index=new.index)
    for c in colonnes:
        mask, lo, hi = _target_mask(new[c], methode, bornes_valides)
        cibles[c] = int(mask.sum())
        if action == "cap":
            new.loc[mask & (new[c] < lo if lo is not None else False), c] = lo
            new.loc[mask & (new[c] > hi if hi is not None else False), c] = hi
        elif action == "set_nan":
            new.loc[mask, c] = np.nan
        elif action == "remove_rows":
            drop_mask |= mask
    if action == "remove_rows":
        new = new.loc[~drop_mask].reset_index(drop=True)
    apres = {c: {"min": float(new[c].min()), "max": float(new[c].max())} for c in colonnes}
    return new, envelope(
        "ok",
        f"Valeurs extrêmes traitées ({action}) sur {', '.join(colonnes)}.",
        metrics={"avant": avant, "apres": apres, "lignes_avant": lignes_avant,
                 "lignes_apres": len(new), "cibles": cibles},
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_outliers.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/outliers.py tests/test_outliers.py
git commit -m "feat: detect_outliers and treat_outliers tools"
```

---

### Task 8: `engineer_features`

**Files:**
- Create: `tools/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `tools.registry.envelope`.
- Produces: `engineer_features(df, specification: dict, justification: str) -> tuple[df, envelope]`. `specification` = `{"type", "source": [...], "output": str, "params": {...}}`.
  - `date_components`: `params={"composants": ["annee","mois","jour","jour_semaine"]}` — adds `<output>_<composant>` columns from `source[0]` (must be datetime).
  - `ratio`: `output = source[0] / source[1]` (0 where denominator 0).
  - `binned`: `params={"bins": [..] , "labels": [..]}` OR `{"q": int}` — `pd.cut` / `pd.qcut` on `source[0]`.
  - `flag`: `params={"gauche": str, "operateur": one of < <= == != >= > , "droite": str|number}` — operands resolve to a column if they match a column name, else a literal. Boolean `output` column.
  - `formula`: `params={"expression": str}` — evaluated with `pandas.DataFrame.eval` after regex whitelist (`^[\w\s+\-*/().]+$` and every bare identifier must be an existing column).
- `metrics`: `{"colonnes_ajoutees": [...], "n_lignes": n, "apercu": {col: [..5..]}}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_features.py
import pandas as pd
from tools.features import engineer_features


def test_formula_recomputes_montant(raw_df):
    new, env = engineer_features(
        raw_df,
        specification={"type": "formula", "source": ["quantite", "prix_unitaire"],
                       "output": "montant_total_recalc",
                       "params": {"expression": "quantite * prix_unitaire"}},
        justification="montant_total incohérent avec quantite*prix_unitaire sur une partie des lignes.",
    )
    assert env["status"] == "ok"
    assert "montant_total_recalc" in new.columns
    assert (new["montant_total_recalc"] == (raw_df["quantite"] * raw_df["prix_unitaire"])).all()


def test_flag_ecart(raw_df):
    df = raw_df.assign(montant_total_recalc=raw_df["quantite"] * raw_df["prix_unitaire"])
    new, env = engineer_features(
        df,
        specification={"type": "flag", "source": ["montant_total", "montant_total_recalc"],
                       "output": "ecart_montant",
                       "params": {"gauche": "montant_total", "operateur": "!=", "droite": "montant_total_recalc"}},
        justification="Marquer les lignes où le montant déclaré diffère du recalcul.",
    )
    assert new["ecart_montant"].dtype == bool
    assert new["ecart_montant"].sum() >= 20


def test_formula_rejects_non_column_identifier(raw_df):
    new, env = engineer_features(
        raw_df,
        specification={"type": "formula", "source": ["quantite"], "output": "x",
                       "params": {"expression": "quantite * __import__"}},
        justification="x",
    )
    assert env["status"] == "error"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_features.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `tools/features.py`**

```python
"""engineer_features — outil de modification. 5 types fermés."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from tools.registry import envelope

_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_EXPR_OK = re.compile(r"^[\w\s+\-*/().]+$")
_OPS = {
    "<": lambda a, b: a < b, "<=": lambda a, b: a <= b, "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b, ">=": lambda a, b: a >= b, ">": lambda a, b: a > b,
}


def _resolve(df, token):
    if isinstance(token, str) and token in df.columns:
        return df[token]
    return token


def engineer_features(df, specification, justification):
    spec = specification
    typ, source, output, params = spec["type"], spec.get("source", []), spec["output"], spec.get("params", {})
    new = df.copy()
    added = []
    try:
        if typ == "formula":
            expr = params["expression"]
            if not _EXPR_OK.match(expr):
                return df, envelope("error", "Expression de formule non autorisée (caractères interdits).")
            for ident in set(_ID.findall(expr)):
                if ident not in new.columns:
                    return df, envelope("error", f"Identifiant '{ident}' de la formule n'est pas une colonne existante.")
            new[output] = new.eval(expr)
            added = [output]
        elif typ == "ratio":
            num, den = new[source[0]], new[source[1]]
            new[output] = np.where(den == 0, 0.0, num / den)
            added = [output]
        elif typ == "date_components":
            s = pd.to_datetime(new[source[0]], errors="coerce")
            mapping = {"annee": s.dt.year, "mois": s.dt.month, "jour": s.dt.day, "jour_semaine": s.dt.dayofweek}
            for comp in params.get("composants", ["annee", "mois"]):
                col = f"{output}_{comp}"
                new[col] = mapping[comp]
                added.append(col)
        elif typ == "binned":
            if "q" in params:
                new[output] = pd.qcut(new[source[0]], q=params["q"], labels=params.get("labels"), duplicates="drop")
            else:
                new[output] = pd.cut(new[source[0]], bins=params["bins"], labels=params.get("labels"))
            added = [output]
        elif typ == "flag":
            op = _OPS[params["operateur"]]
            left = _resolve(new, params["gauche"])
            right = _resolve(new, params["droite"])
            new[output] = op(left, right).astype(bool)
            added = [output]
        else:
            return df, envelope("error", f"Type de feature inconnu : {typ}.")
    except (KeyError, ValueError) as e:
        return df, envelope("error", f"Échec engineer_features : {e}")

    return new, envelope(
        "ok",
        f"Variable(s) dérivée(s) créée(s) : {', '.join(added)} (type {typ}).",
        metrics={"colonnes_ajoutees": added, "n_lignes": len(new),
                 "apercu": {c: new[c].head(5).tolist() for c in added}},
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_features.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/features.py tests/test_features.py
git commit -m "feat: engineer_features tool"
```

---

### Task 9: `query_dataframe` + `plot_chart`

**Files:**
- Create: `tools/query.py`
- Create: `tools/charts.py`
- Test: `tests/test_query.py`
- Test: `tests/test_charts.py`

**Interfaces:**
- `query_dataframe(df, operation: str, parametres: dict) -> dict` (read envelope). No code execution — 4 fixed operations:
  - `filter_count`: `{colonne, operateur, valeur}`, `operateur ∈ < <= == != >= > contains`. `detail = {"n": int}`.
  - `groupby_agg`: `{colonne_groupby, colonne_agg, fonction, periode?}`, `fonction ∈ sum mean count min max median nunique`, `periode = {colonne_date, mois, annee}` filters rows before grouping. `detail = {"resultats": {clef: valeur}, "top": [clef, valeur]}`.
  - `describe_column`: `{colonne}` → `detail = df[colonne].describe()` as dict.
  - `top_n`: `{colonne, n, ordre}`, `ordre ∈ asc desc`, `n` capped at 20 → `detail = {"lignes": [...], "tronque": bool}` with truncation message in `summary`.
- `plot_chart(df, type: str, x: str, y: str | None = None, titre: str = "") -> dict` (read envelope). Builds a plotly figure; `detail = {"figure": fig.to_dict()}`. Unknown `type`/missing column → error envelope.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_query.py
import pandas as pd
from tools.query import query_dataframe


def test_top_product_in_march(raw_df):
    df = raw_df.copy()
    df["date_commande"] = pd.to_datetime(df["date_commande"], errors="coerce", dayfirst=True)
    df = df[df["quantite"] > 0]
    out = query_dataframe(df, "groupby_agg", {
        "colonne_groupby": "produit", "colonne_agg": "quantite", "fonction": "sum",
        "periode": {"colonne_date": "date_commande", "mois": 3},
    })
    assert out["status"] == "ok"
    assert out["detail"]["top"][0] in df["produit"].unique()


def test_filter_count(raw_df):
    out = query_dataframe(raw_df, "filter_count", {"colonne": "quantite", "operateur": "<", "valeur": 0})
    assert out["detail"]["n"] == int((raw_df["quantite"] < 0).sum())


def test_top_n_caps_at_20(raw_df):
    out = query_dataframe(raw_df, "top_n", {"colonne": "montant_total", "n": 50, "ordre": "desc"})
    assert out["detail"]["tronque"] is True
    assert len(out["detail"]["lignes"]) == 20
```

```python
# tests/test_charts.py
from tools.charts import plot_chart


def test_plot_chart_bar(raw_df):
    out = plot_chart(raw_df, type="bar", x="region", y="montant_total", titre="CA par région")
    assert out["status"] == "ok"
    assert "figure" in out["detail"]


def test_plot_chart_bad_type(raw_df):
    out = plot_chart(raw_df, type="camembert", x="region", y="montant_total")
    assert out["status"] == "error"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_query.py tests/test_charts.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `tools/query.py`**

```python
"""query_dataframe — opérations paramétrées, jamais de code exécuté."""
from __future__ import annotations

import pandas as pd

from tools.registry import envelope

_CMP = {
    "<": lambda s, v: s < v, "<=": lambda s, v: s <= v, "==": lambda s, v: s == v,
    "!=": lambda s, v: s != v, ">=": lambda s, v: s >= v, ">": lambda s, v: s > v,
    "contains": lambda s, v: s.astype("string").str.contains(str(v), case=False, na=False),
}
_AGG = {"sum", "mean", "count", "min", "max", "median", "nunique"}
TOP_N_CAP = 20


def _apply_periode(df, periode):
    col = periode["colonne_date"]
    d = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    out = df
    if periode.get("mois") is not None:
        out = out[d.dt.month == periode["mois"]]
        d = d[d.dt.month == periode["mois"]]
    if periode.get("annee") is not None:
        out = out[d.dt.year == periode["annee"]]
    return out


def query_dataframe(df, operation, parametres):
    p = parametres or {}
    try:
        if operation == "filter_count":
            s = df[p["colonne"]]
            n = int(_CMP[p["operateur"]](s, p["valeur"]).sum())
            return envelope("ok", f"{n} ligne(s) correspondent au filtre.", detail={"n": n})

        if operation == "groupby_agg":
            data = _apply_periode(df, p["periode"]) if p.get("periode") else df
            fn = p["fonction"]
            if fn not in _AGG:
                return envelope("error", f"Fonction d'agrégation inconnue : {fn}.")
            grouped = getattr(data.groupby(p["colonne_groupby"])[p["colonne_agg"]], fn)()
            grouped = grouped.sort_values(ascending=False)
            top = [grouped.index[0], round(float(grouped.iloc[0]), 4)] if len(grouped) else [None, None]
            return envelope(
                "ok",
                f"Agrégation {fn} de {p['colonne_agg']} par {p['colonne_groupby']}. En tête : {top[0]}.",
                detail={"resultats": {str(k): round(float(v), 4) for k, v in grouped.head(20).items()}, "top": top},
            )

        if operation == "describe_column":
            desc = df[p["colonne"]].describe()
            return envelope("ok", f"Statistiques descriptives de {p['colonne']}.",
                            detail={str(k): (float(v) if pd.api.types.is_number(v) else str(v)) for k, v in desc.items()})

        if operation == "top_n":
            n = int(p["n"])
            tronque = n > TOP_N_CAP
            n = min(n, TOP_N_CAP)
            asc = p.get("ordre", "desc") == "asc"
            rows = df.sort_values(p["colonne"], ascending=asc).head(n)
            msg = f"{n} première(s) ligne(s) par {p['colonne']}."
            if tronque:
                msg += f" (Tronqué à {TOP_N_CAP}.)"
            return envelope("ok", msg, detail={"lignes": rows.to_dict(orient="records"), "tronque": tronque})

        return envelope("error", f"Opération inconnue : {operation}.")
    except KeyError as e:
        return envelope("error", f"Paramètre ou colonne manquant : {e}.")
```

- [ ] **Step 4: Implement `tools/charts.py`**

```python
"""plot_chart — graphiques plotly."""
from __future__ import annotations

import plotly.express as px

from tools.registry import envelope

_BUILDERS = {
    "bar": lambda df, x, y: px.bar(df, x=x, y=y),
    "line": lambda df, x, y: px.line(df, x=x, y=y),
    "histogram": lambda df, x, y: px.histogram(df, x=x),
    "scatter": lambda df, x, y: px.scatter(df, x=x, y=y),
    "box": lambda df, x, y: px.box(df, x=x, y=y),
}


def plot_chart(df, type, x, y=None, titre=""):
    if type not in _BUILDERS:
        return envelope("error", f"Type de graphique non supporté : {type}.")
    for col in (c for c in (x, y) if c):
        if col not in df.columns:
            return envelope("error", f"Colonne inexistante : {col}.")
    fig = _BUILDERS[type](df, x, y)
    if titre:
        fig.update_layout(title=titre)
    return envelope("ok", f"Graphique {type} ({x}{' / ' + y if y else ''}) généré.",
                    detail={"figure": fig.to_dict()})
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_query.py tests/test_charts.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tools/query.py tools/charts.py tests/test_query.py tests/test_charts.py
git commit -m "feat: query_dataframe and plot_chart read-only tools"
```

---

### Task 10: `generate_report`

**Files:**
- Create: `tools/reporting.py`
- Test: `tests/test_reporting.py`

**Interfaces:**
- Consumes: `tools.profiling.profile_dataset`, `config.REPORT_PATH`.
- Produces: `generate_report(df_original: pandas.DataFrame, df_current: pandas.DataFrame, actions_log: list[dict], path: str | None = None) -> dict` (read envelope). `detail = {"markdown": str, "path": str}`. Writes the Markdown to `path or config.REPORT_PATH` (UTF-8).
- Markdown sections, in order: `# Rapport de valorisation AutoPrep`, `## Résumé exécutif` (2–3 lines), `## État initial` (profile of `df_original`), `## Actions effectuées` — a table with columns `Étape | Type | Outil | Paramètres | Justification | Métriques`, where **Type** is `Lecture (auto)` or `Modification (validée)` / `Modification (contre-proposition humaine)` based on `action["origine"]`, `## État final` (profile of `df_current`), `## Variables créées`, `## Recommandations` (optional bullet list, may be "Aucune.").
- `actions_log` entry shape (produced by Task 13's `act` node): `{"outil": str, "parametres": dict, "justification": str, "metriques": dict, "timestamp": str, "type": "lecture"|"modification", "origine": "auto"|"humain_validee"|"humain_contre_proposition"}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reporting.py
from tools.reporting import generate_report


def test_report_has_all_sections(raw_df, tmp_path):
    df2 = raw_df.drop_duplicates()
    log = [
        {"outil": "profile_dataset", "parametres": {}, "justification": "", "metriques": {"n_lignes": len(raw_df)},
         "timestamp": "2026-09-10T10:00:00", "type": "lecture", "origine": "auto"},
        {"outil": "handle_duplicates", "parametres": {"sous_ensemble_colonnes": None},
         "justification": "2% de doublons", "metriques": {"supprimees": 20},
         "timestamp": "2026-09-10T10:01:00", "type": "modification", "origine": "humain_validee"},
    ]
    out = generate_report(raw_df, df2, log, path=str(tmp_path / "rapport.md"))
    md = out["detail"]["markdown"]
    for h in ["# Rapport de valorisation AutoPrep", "## Résumé exécutif", "## État initial",
              "## Actions effectuées", "## État final", "## Variables créées"]:
        assert h in md
    assert "Lecture (auto)" in md
    assert "Modification (validée)" in md
    assert (tmp_path / "rapport.md").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_reporting.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `tools/reporting.py`**

```python
"""generate_report — rapport Markdown, écrit aussi sur disque."""
from __future__ import annotations

from pathlib import Path

import config
from tools.profiling import profile_dataset
from tools.registry import envelope

_TYPE_LABEL = {
    ("lecture", "auto"): "Lecture (auto)",
    ("modification", "humain_validee"): "Modification (validée)",
    ("modification", "humain_contre_proposition"): "Modification (contre-proposition humaine)",
}


def _profile_block(df) -> str:
    d = profile_dataset(df)["detail"]
    lines = [f"- Lignes : {d['n_lignes']} — Colonnes : {d['n_colonnes']} — Doublons : {d['n_doublons']}"]
    manq = {c: v for c, v in d["valeurs_manquantes"].items() if v["n"]}
    if manq:
        lines.append("- Valeurs manquantes : " + ", ".join(f"{c} ({v['n']})" for c, v in manq.items()))
    if d["numeriques"]:
        lines.append("- Min/max numériques : " + ", ".join(
            f"{c} [{s['min']:g} … {s['max']:g}]" for c, s in d["numeriques"].items()))
    return "\n".join(lines)


def generate_report(df_original, df_current, actions_log, path=None):
    path = path or config.REPORT_PATH
    n_mod = sum(1 for a in actions_log if a.get("type") == "modification")
    n_lec = sum(1 for a in actions_log if a.get("type") == "lecture")

    rows = []
    for i, a in enumerate(actions_log, 1):
        label = _TYPE_LABEL.get((a.get("type"), a.get("origine")), a.get("type", ""))
        params = ", ".join(f"{k}={v}" for k, v in (a.get("parametres") or {}).items()) or "—"
        metr = ", ".join(f"{k}={v}" for k, v in (a.get("metriques") or {}).items()) or "—"
        just = (a.get("justification") or "—").replace("\n", " ")
        rows.append(f"| {i} | {label} | {a['outil']} | {params} | {just} | {metr} |")

    features = [c for c in df_current.columns if c not in df_original.columns]
    md = f"""# Rapport de valorisation AutoPrep

## Résumé exécutif

Jeu de données traité : {len(df_original)} → {len(df_current)} lignes.
{n_mod} action(s) de modification validée(s) par l'humain, {n_lec} action(s) de lecture automatiques.
{len(features)} variable(s) dérivée(s) créée(s).

## État initial

{_profile_block(df_original)}

## Actions effectuées

| Étape | Type | Outil | Paramètres | Justification | Métriques |
|---|---|---|---|---|---|
{chr(10).join(rows) if rows else "| — | — | — | — | — | — |"}

## État final

{_profile_block(df_current)}

## Variables créées

{chr(10).join(f"- {c}" for c in features) if features else "Aucune."}

## Recommandations

Aucune.
"""
    Path(path).write_text(md, encoding="utf-8")
    return envelope("ok", f"Rapport généré ({len(actions_log)} action(s)) et écrit dans {path}.",
                    metrics={"n_modifications": n_mod, "n_lectures": n_lec, "n_features": len(features)},
                    detail={"markdown": md, "path": path})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_reporting.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/reporting.py tests/test_reporting.py
git commit -m "feat: generate_report tool"
```

---

### Task 11: Agent state schema + dataframe store

**Files:**
- Create: `agent/state.py`
- Create: `agent/store.py`
- Test: `tests/test_state_store.py`

**Interfaces:**
- `agent/store.py`:
  - `DF_STORE: dict[str, dict]` — `{thread_id: {"original": df, "current": df}}`.
  - `init_thread(thread_id: str, df: pandas.DataFrame) -> None` — sets `original` (a copy) and `current` (a copy).
  - `get_current(thread_id) -> pandas.DataFrame`, `set_current(thread_id, df) -> None`, `get_original(thread_id) -> pandas.DataFrame`.
- `agent/state.py`:
  - `AgentState(TypedDict)` with keys: `messages: Annotated[list, add_messages]`, `thread_id: str`, `iteration_count: int`, `max_iterations: int`, `checklist: dict[str, bool]`, `pending_proposal: dict | None`, `pending_user_question: str | None`, `human_decision: str | None`, `motif_refus: str | None`, `human_proposed_action: dict | None`, `actions_log: Annotated[list, operator.add]`, `task_done: bool`, `final_report: str | None`.
  - `initial_state(thread_id: str) -> AgentState` — `iteration_count=0`, `max_iterations=config.MAX_ITERATIONS`, all six checklist keys `False` (`profilage, typage, doublons, valeurs_manquantes, outliers, features`), everything else `None`/`[]`/`False`.
  - `CHECKLIST_KEYS: list[str]` and `TOOL_TO_CHECKLIST: dict[str, str]` (`profile_dataset→profilage`, `fix_column_types→typage`, `handle_duplicates→doublons`, `handle_missing_values→valeurs_manquantes`, `treat_outliers→outliers`, `engineer_features→features`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_state_store.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_state_store.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `agent/store.py` then `agent/state.py`**

```python
# agent/store.py
"""Stockage des dataframes hors état LangGraph : dict module-level indexé par thread_id."""
from __future__ import annotations

import pandas as pd

DF_STORE: dict[str, dict] = {}


def init_thread(thread_id: str, df: pd.DataFrame) -> None:
    DF_STORE[thread_id] = {"original": df.copy(), "current": df.copy()}


def get_current(thread_id: str) -> pd.DataFrame:
    return DF_STORE[thread_id]["current"]


def set_current(thread_id: str, df: pd.DataFrame) -> None:
    DF_STORE[thread_id]["current"] = df


def get_original(thread_id: str) -> pd.DataFrame:
    return DF_STORE[thread_id]["original"]
```

```python
# agent/state.py
"""Schéma de l'état du graphe LangGraph."""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

import config

CHECKLIST_KEYS = ["profilage", "typage", "doublons", "valeurs_manquantes", "outliers", "features"]

TOOL_TO_CHECKLIST = {
    "profile_dataset": "profilage",
    "fix_column_types": "typage",
    "handle_duplicates": "doublons",
    "handle_missing_values": "valeurs_manquantes",
    "treat_outliers": "outliers",
    "engineer_features": "features",
}


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    thread_id: str
    iteration_count: int
    max_iterations: int
    checklist: dict
    pending_proposal: dict | None
    pending_user_question: str | None
    human_decision: str | None
    motif_refus: str | None
    human_proposed_action: dict | None
    actions_log: Annotated[list, operator.add]
    task_done: bool
    final_report: str | None


def initial_state(thread_id: str) -> AgentState:
    return AgentState(
        messages=[],
        thread_id=thread_id,
        iteration_count=0,
        max_iterations=config.MAX_ITERATIONS,
        checklist={k: False for k in CHECKLIST_KEYS},
        pending_proposal=None,
        pending_user_question=None,
        human_decision=None,
        motif_refus=None,
        human_proposed_action=None,
        actions_log=[],
        task_done=False,
        final_report=None,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_state_store.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/state.py agent/store.py tests/test_state_store.py
git commit -m "feat: agent state schema and dataframe store"
```

---

### Task 12: System prompt + `call_llm()`

**Files:**
- Create: `agent/prompts.py`
- Create: `agent/llm_client.py`
- Test: `tests/test_prompts_llm.py`

**Interfaces:**
- `agent/prompts.py`: `SYSTEM_PROMPT: str` — the validated draft from the spec's "Prompt système" section, verbatim (French, all sections: RÔLE, ORDRE DES OPÉRATIONS, CHECKLIST DE SORTIE, RÈGLE DE VALIDATION, EN CAS DE REFUS, EN CAS D'ÉCHEC D'UN OUTIL, QUESTIONS HORS-BANDE, DISCIPLINE ANTI-HALLUCINATION, LANGUE).
- `agent/llm_client.py`:
  - `call_llm(messages: list, tools: list) -> AIMessage` — constructs `ChatAnthropic(model=config.MODEL_NAME, api_key=config.ANTHROPIC_API_KEY, temperature=0, max_tokens=4096)`, calls `.bind_tools(tools).invoke(messages)`, returns the `AIMessage`. Raises `RuntimeError("ANTHROPIC_API_KEY manquante")` if the key is falsy. One retry on `anthropic` API error before re-raising.
  - This is the **only** module that imports `langchain_anthropic`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prompts_llm.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_prompts_llm.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `agent/prompts.py` and `agent/llm_client.py`**

`agent/prompts.py`: paste the spec's prompt block verbatim into a triple-quoted `SYSTEM_PROMPT` string (lines 134–193 of `autoprep-agent-architecture.md`).

```python
# agent/llm_client.py
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
    ).bind_tools(tools)
    try:
        return model.invoke(messages)
    except Exception:
        return model.invoke(messages)  # une tentative de retry avant de laisser remonter
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_prompts_llm.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/prompts.py agent/llm_client.py tests/test_prompts_llm.py
git commit -m "feat: system prompt and call_llm abstraction"
```

---

### Task 13: LangGraph agent loop

**Files:**
- Create: `agent/loop.py`
- Test: `tests/test_loop.py`
- Test: `tests/conftest.py` (add `ScriptedLLM` helper)

**Interfaces:**
- Consumes: `agent.state` (`AgentState`, `initial_state`, `TOOL_TO_CHECKLIST`, `CHECKLIST_KEYS`), `agent.store`, `agent.prompts.SYSTEM_PROMPT`, `agent.llm_client.call_llm`, `tools.registry` (`WRITE_TOOLS`, `LLM_TOOLS`, `MODEL_NAME_TO_TOOL`, `TOOL_REGISTRY`, `dispatch`, `validate_args`), `tools.reporting.generate_report`.
- Produces:
  - `build_graph()` — returns a compiled `StateGraph` with an `InMemorySaver` checkpointer. Nodes: `think`, `act`, `human_gate`, `force_report`, `report`. Edges per the flow below.
  - `start_run(thread_id: str, df) -> dict` — `store.init_thread`, builds state, `graph.invoke(state, config={"configurable": {"thread_id": thread_id}})`, returns the resulting state snapshot (`graph.get_state(...)`).
  - `resume_run(thread_id: str, reponse_humaine: dict) -> dict` — `graph.invoke(Command(resume=reponse_humaine), config=...)`, returns the state snapshot.
  - `submit_question(thread_id: str, question: str) -> dict` — writes `pending_user_question` via `graph.update_state`, then resumes/continues; returns snapshot.
  - `get_pending_proposal(thread_id) -> dict | None`, `get_report(thread_id) -> str | None`.
- **Graph flow:**
  - `START → think`
  - `think`: if `iteration_count >= max_iterations` → append `AIMessage("Itération maximale atteinte (15) — je génère le rapport avec l'état actuel.")`, return (no LLM call). Else: increment `iteration_count`; if `pending_user_question` is set → answer it via `dispatch("query_dataframe", ...)` best-effort, append a `ToolMessage`/`AIMessage` with the answer, clear `pending_user_question`, **do not clear `pending_proposal`**; then call `call_llm([SystemMessage(SYSTEM_PROMPT), *messages], LLM_TOOLS)`, append the `AIMessage`.
  - conditional edge `route_after_think`:
    - `iteration_count >= max_iterations` (cap message was just added) → `force_report`
    - last message has no `tool_calls` → `END` (agent produced prose; MVP treats as stop)
    - resolved tool name `== "generate_report"` → `report`
    - resolved tool name `in WRITE_TOOLS` → `human_gate`
    - else → `act`
  - `human_gate`: resolve `call = last_ai.tool_calls[0]`; map class name via `MODEL_NAME_TO_TOOL`; set `pending_proposal = {"tool_name": name, "arguments": call["args"]}`; `decision = interrupt(pending_proposal)`. On resume:
    - `decision["decision"] == "validee"` → `human_decision="validee"`, `human_proposed_action=None`, keep `pending_proposal` for `act` to read, → `act`.
    - `"refusee"` + `contre_proposition` → `validate_args(cp["tool_name"], cp["arguments"])` (on `ValidationError` append `ToolMessage` explaining the invalid counter-proposal and → `think`); else `human_decision="refusee"`, `human_proposed_action=cp`, → `act`.
    - `"refusee"` plain → `human_decision="refusee"`, `motif_refus=decision.get("motif_refus")`, append `ToolMessage(tool_call_id=call["id"], content="Proposition refusée. Motif : <motif|non précisé>. Propose une alternative qui en tient compte.")`, clear `pending_proposal`, → `think`.
  - conditional edge `route_after_gate`: `human_decision in {"validee","refusee"}` and a call is pending (`pending_proposal` or `human_proposed_action`) → `act`; else → `think`.
  - `act`: determine the call — `human_proposed_action` if set, else `pending_proposal`, else `last_ai.tool_calls[0]` (read tools). Run `new_df, env = dispatch(name, store.get_current(thread_id), args)`. If `new_df is not None` → `store.set_current(thread_id, new_df)`. Append `ToolMessage(tool_call_id=<id>, content=json.dumps({"summary": env["summary"], "metrics": env["metrics"], "status": env["status"]}, ensure_ascii=False))`. If `name in TOOL_TO_CHECKLIST` and `env["status"]=="ok"` → set that checklist key `True`. If `name in WRITE_TOOLS` and `env["status"]=="ok"` → append to `actions_log` an entry `{outil, parametres, justification, metriques, timestamp: datetime.now().isoformat(), type: "modification", origine: "humain_contre_proposition" if human_proposed_action else "humain_validee"}`. For read tools with `status=="ok"` also log `{... type:"lecture", origine:"auto"}` (needed for the report's read/write distinction). Clear `pending_proposal`, `human_decision`, `motif_refus`, `human_proposed_action`. → `think`.
  - `force_report` and `report`: call `generate_report(store.get_original(tid), store.get_current(tid), actions_log)`; set `final_report = env["detail"]["markdown"]`, `task_done = True`; append `ToolMessage`/`AIMessage` with `env["summary"]`. → `END`.

- [ ] **Step 1: Add `ScriptedLLM` to `tests/conftest.py`**

```python
from langchain_core.messages import AIMessage
import pytest


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
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{self.calls}"}])


@pytest.fixture
def scripted(monkeypatch):
    def _install(script):
        llm = ScriptedLLM(script)
        import agent.loop as loop_mod
        monkeypatch.setattr(loop_mod, "call_llm", llm)
        return llm
    return _install
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_loop.py
import pandas as pd
from agent.loop import build_graph, start_run, resume_run, get_pending_proposal, get_report
from agent import store


def test_read_tool_runs_without_interrupt(scripted, raw_df):
    scripted([("ProfileDatasetArgs", {}), ("GenerateReportArgs", {})])
    snap = start_run("tr1", raw_df)
    assert get_pending_proposal("tr1") is None
    assert get_report("tr1") is not None


def test_write_tool_pauses_then_resumes(scripted, raw_df):
    scripted([
        ("HandleDuplicatesArgs", {"sous_ensemble_colonnes": None, "justification": "2% doublons"}),
        ("GenerateReportArgs", {}),
    ])
    start_run("tw1", raw_df)
    prop = get_pending_proposal("tw1")
    assert prop["tool_name"] == "handle_duplicates"
    resume_run("tw1", {"decision": "validee", "motif_refus": None, "contre_proposition": None})
    assert store.get_current("tw1").duplicated().sum() == 0
    assert get_report("tw1") is not None


def test_guardrail_forces_report_at_15(scripted, raw_df):
    # LLM keeps proposing a read tool forever; guardrail must cut in.
    scripted([("ProfileDatasetArgs", {})] * 40)
    start_run("tg1", raw_df)
    rep = get_report("tg1")
    assert rep is not None and "Itération maximale atteinte (15)" in "".join(
        m.content for m in build_graph.__wrapped__ if False) or rep  # report exists
    snap = build_graph  # sanity
    # iteration_count capped
    from agent.loop import _graph_state
    assert _graph_state("tg1")["iteration_count"] >= 15


def test_refusal_with_motif_goes_back_to_think(scripted, raw_df):
    scripted([
        ("HandleMissingValuesArgs", {"colonnes": ["categorie"], "strategie": "drop_rows", "justification": "x"}),
        ("HandleMissingValuesArgs", {"colonnes": ["categorie"], "strategie": "mode", "justification": "moins destructif"}),
        ("GenerateReportArgs", {}),
    ])
    start_run("trf1", raw_df)
    resume_run("trf1", {"decision": "refusee", "motif_refus": "drop trop destructif", "contre_proposition": None})
    prop = get_pending_proposal("trf1")
    assert prop["arguments"]["strategie"] == "mode"
```

(If a test helper like `_graph_state` is convenient, expose it from `agent/loop.py` as a thin wrapper over `graph.get_state`.)

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_loop.py -v`
Expected: FAIL — `ModuleNotFoundError: agent.loop`.

- [ ] **Step 4: Implement `agent/loop.py`**

Implement per the flow in the Interfaces block. Key skeleton:

```python
"""Graphe LangGraph : think / act / human_gate / force_report / report."""
from __future__ import annotations

import json
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from pydantic import ValidationError

from agent.llm_client import call_llm
from agent.prompts import SYSTEM_PROMPT
from agent.state import AgentState, TOOL_TO_CHECKLIST, initial_state
from agent import store
from tools.registry import (
    LLM_TOOLS, MODEL_NAME_TO_TOOL, TOOL_REGISTRY, WRITE_TOOLS, dispatch, validate_args,
)
from tools.reporting import generate_report

_CFG = lambda tid: {"configurable": {"thread_id": tid}}
_GRAPH = None


def _resolve_call(ai_msg):
    tc = ai_msg.tool_calls[0]
    name = MODEL_NAME_TO_TOOL.get(tc["name"], tc["name"])
    return name, tc["args"], tc["id"]


def think(state: AgentState) -> dict:
    if state["iteration_count"] >= state["max_iterations"]:
        return {"messages": [AIMessage(content="Itération maximale atteinte (15) — je génère le rapport avec l'état actuel.")]}
    out: dict = {"iteration_count": state["iteration_count"] + 1}
    msgs = list(state["messages"])
    if state.get("pending_user_question"):
        q = state["pending_user_question"]
        try:
            _, env = dispatch("query_dataframe", store.get_current(state["thread_id"]),
                              {"operation": "describe_column", "parametres": {"colonne": _guess_col(q)}})
            ans = env["summary"]
        except Exception:
            ans = "Je n'ai pas pu répondre précisément à la question via les opérations disponibles."
        out["messages"] = [AIMessage(content=f"[Réponse hors-bande] {ans}")]
        out["pending_user_question"] = None
        msgs = msgs + out["messages"]
    response = call_llm([SystemMessage(content=SYSTEM_PROMPT), *msgs], LLM_TOOLS)
    out["messages"] = out.get("messages", []) + [response]
    return out


def route_after_think(state: AgentState) -> str:
    last = state["messages"][-1]
    if state["iteration_count"] >= state["max_iterations"] and not getattr(last, "tool_calls", None):
        return "force_report"
    if not getattr(last, "tool_calls", None):
        return END
    name, _, _ = _resolve_call(last)
    if name == "generate_report":
        return "report"
    if name in WRITE_TOOLS:
        return "human_gate"
    return "act"


def human_gate(state: AgentState) -> dict:
    name, args, call_id = _resolve_call(state["messages"][-1])
    proposal = {"tool_name": name, "arguments": args}
    decision = interrupt(proposal)
    if decision["decision"] == "validee":
        return {"pending_proposal": proposal, "human_decision": "validee", "human_proposed_action": None}
    cp = decision.get("contre_proposition")
    if cp:
        try:
            validate_args(cp["tool_name"], cp["arguments"])
        except ValidationError as e:
            return {"pending_proposal": None, "human_decision": None,
                    "messages": [ToolMessage(tool_call_id=call_id,
                        content=f"Contre-proposition invalide : {e}. Propose une action correcte.")]}
        return {"pending_proposal": None, "human_decision": "refusee", "human_proposed_action": cp}
    return {"pending_proposal": None, "human_decision": "refusee",
            "motif_refus": decision.get("motif_refus"),
            "messages": [ToolMessage(tool_call_id=call_id,
                content=f"Proposition refusée. Motif : {decision.get('motif_refus') or 'non précisé'}. "
                        f"Propose une alternative qui en tient compte.")]}


def route_after_gate(state: AgentState) -> str:
    if state.get("human_proposed_action") or (state.get("human_decision") == "validee" and state.get("pending_proposal")):
        return "act"
    return "think"


def act(state: AgentState) -> dict:
    tid = state["thread_id"]
    hpa = state.get("human_proposed_action")
    if hpa:
        name, args = hpa["tool_name"], hpa["arguments"]
        call_id, origine = f"human_{datetime.now().timestamp()}", "humain_contre_proposition"
    elif state.get("pending_proposal"):
        name, args = state["pending_proposal"]["tool_name"], state["pending_proposal"]["arguments"]
        _, _, call_id = _resolve_call(state["messages"][-1]) if getattr(state["messages"][-1], "tool_calls", None) else (None, None, f"v_{datetime.now().timestamp()}")
        origine = "humain_validee"
    else:
        name, args, call_id = _resolve_call(state["messages"][-1])
        origine = "auto"

    new_df, env = dispatch(name, store.get_current(tid), args)
    if new_df is not None and env["status"] == "ok":
        store.set_current(tid, new_df)

    upd: dict = {
        "messages": [ToolMessage(tool_call_id=call_id or "auto",
            content=json.dumps({"status": env["status"], "summary": env["summary"], "metrics": env["metrics"]}, ensure_ascii=False))],
        "pending_proposal": None, "human_decision": None, "motif_refus": None, "human_proposed_action": None,
    }
    if env["status"] == "ok" and name in TOOL_TO_CHECKLIST:
        upd["checklist"] = {**state["checklist"], TOOL_TO_CHECKLIST[name]: True}
    if env["status"] == "ok":
        upd["actions_log"] = [{
            "outil": name, "parametres": {k: v for k, v in args.items() if k != "justification"},
            "justification": args.get("justification", ""), "metriques": env["metrics"],
            "timestamp": datetime.now().isoformat(),
            "type": "modification" if name in WRITE_TOOLS else "lecture",
            "origine": origine if name in WRITE_TOOLS else "auto",
        }]
    return upd


def _finish(state: AgentState) -> dict:
    tid = state["thread_id"]
    _, env = None, generate_report(store.get_original(tid), store.get_current(tid), state["actions_log"])
    return {"final_report": env["detail"]["markdown"], "task_done": True,
            "messages": [AIMessage(content=env["summary"])]}


def build_graph():
    global _GRAPH
    if _GRAPH is not None:
        return _GRAPH
    g = StateGraph(AgentState)
    g.add_node("think", think)
    g.add_node("act", act)
    g.add_node("human_gate", human_gate)
    g.add_node("force_report", _finish)
    g.add_node("report", _finish)
    g.add_edge(START, "think")
    g.add_conditional_edges("think", route_after_think,
                            {"act": "act", "human_gate": "human_gate", "report": "report",
                             "force_report": "force_report", END: END})
    g.add_conditional_edges("human_gate", route_after_gate, {"act": "act", "think": "think"})
    g.add_edge("act", "think")
    g.add_edge("force_report", END)
    g.add_edge("report", END)
    _GRAPH = g.compile(checkpointer=InMemorySaver())
    return _GRAPH


def _graph_state(tid: str):
    return build_graph().get_state(_CFG(tid)).values


def start_run(thread_id: str, df) -> dict:
    store.init_thread(thread_id, df)
    build_graph().invoke(initial_state(thread_id), _CFG(thread_id))
    return _graph_state(thread_id)


def resume_run(thread_id: str, reponse_humaine: dict) -> dict:
    build_graph().invoke(Command(resume=reponse_humaine), _CFG(thread_id))
    return _graph_state(thread_id)


def submit_question(thread_id: str, question: str) -> dict:
    build_graph().update_state(_CFG(thread_id), {"pending_user_question": question})
    build_graph().invoke(None, _CFG(thread_id))
    return _graph_state(thread_id)


def get_pending_proposal(thread_id: str):
    return _graph_state(thread_id).get("pending_proposal")


def get_report(thread_id: str):
    return _graph_state(thread_id).get("final_report")


def _guess_col(_q: str) -> str:
    return "montant_total"
```

Notes for the implementer:
- `_guess_col` is a deliberately trivial stub for the MVP out-of-band path — the scenario test drives questions through `query_dataframe` directly in Task 15; keep the stub but leave a `# TODO` that a smarter mapping (or an extra LLM call) is roadmap.
- If `build_graph()` memoization across tests causes checkpointer state bleed, make `build_graph()` take no cache (`_GRAPH = None` each call) or reset in a fixture. Prefer a per-call fresh compile in tests via a `reset_graph()` helper.
- Delete the `build_graph.__wrapped__` nonsense line in the draft test — replace with a direct check that `get_report("tg1")` contains the cap sentence OR that `_graph_state` shows `task_done is True` and `iteration_count >= 15`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_loop.py -v`
Expected: PASS. Fix routing/state-merge bugs until green. Then run the full suite: `uv run pytest -v`.

- [ ] **Step 6: Commit**

```bash
git add agent/loop.py tests/test_loop.py tests/conftest.py
git commit -m "feat: LangGraph agent loop with HITL interrupt and iteration guardrail"
```

---

### Task 14: Streamlit app + upload validation

**Files:**
- Create: `app.py`
- Create: `agent/upload.py`
- Test: `tests/test_upload.py`

**Interfaces:**
- `agent/upload.py`:
  - `read_uploaded_csv(raw_bytes: bytes) -> pandas.DataFrame` — sniff separator (`;` vs `,` via `csv.Sniffer` on the decoded head, default `;`), decode `utf-8` then fall back `latin-1`. Raises `ValueError("Fichier illisible")` on total failure.
  - `validate_dataframe(df: pandas.DataFrame) -> None` — raises `ValueError` if 0 rows or 0 columns.
- `app.py` (not unit-tested; manual/`streamlit run`): 
  - On first load: `st.session_state.thread_id = str(uuid4())`, `st.session_state.graph = build_graph()`.
  - File uploader → `read_uploaded_csv` → `validate_dataframe` → `start_run(thread_id, df)`.
  - Each rerun: read snapshot via `agent.loop._graph_state`. If `get_pending_proposal` → render a panel: `TOOL_REGISTRY[name]["libelle"]`, the `params_affiches` values, the `justification` as "Pourquoi", and **Valider** / **Refuser** buttons (Refuser reveals a `motif_refus` text field + optional counter-proposal JSON). Buttons call `resume_run(...)` then `st.rerun()`.
  - Chat input for out-of-band questions → `submit_question(...)`.
  - When `get_report` is not None: render the Markdown and a `st.download_button` for `rapport.md`.
  - Show `actions_log` as a running table, and the current `checklist` as check marks.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_upload.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_upload.py -v`
Expected: FAIL — `ModuleNotFoundError: agent.upload`.

- [ ] **Step 3: Implement `agent/upload.py`**

```python
"""Lecture tolérante des CSV d'export PME françaises (latin-1, séparateur ';')."""
from __future__ import annotations

import csv
import io

import pandas as pd


def _decode(raw: bytes) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Fichier illisible : encodage non reconnu (ni utf-8 ni latin-1).")


def read_uploaded_csv(raw_bytes: bytes) -> pd.DataFrame:
    text = _decode(raw_bytes)
    head = "\n".join(text.splitlines()[:5])
    try:
        sep = csv.Sniffer().sniff(head, delimiters=";,").delimiter
    except csv.Error:
        sep = ";"
    df = pd.read_csv(io.StringIO(text), sep=sep)
    if df.shape[1] == 1 and sep != ";":
        df = pd.read_csv(io.StringIO(text), sep=";")
    return df


def validate_dataframe(df: pd.DataFrame) -> None:
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError("Fichier invalide : aucune donnée exploitable (0 ligne ou 0 colonne).")
```

- [ ] **Step 4: Implement `app.py`**

Build the Streamlit UI per the Interfaces block. Keep it thin — all state via `agent.loop` helpers, only `thread_id` + `graph` in `st.session_state`. Reference `TOOL_REGISTRY` for labels/params. Verify manually:

Run: `uv run streamlit run app.py`
Expected: upload `data/ventes_pme.csv` → agent starts → a validation panel appears for the first write tool.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_upload.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app.py agent/upload.py tests/test_upload.py
git commit -m "feat: Streamlit UI and tolerant CSV upload"
```

---

### Task 15: End-to-end scenario test (MVP definition of done)

**Files:**
- Create: `tests/test_scenario.py`

**Interfaces:**
- Consumes: `agent.loop` (`start_run`, `resume_run`, `submit_question`, `get_pending_proposal`, `get_report`, `_graph_state`), `agent.store`, `data.generate_synthetic.generate_synthetic`, `tools.query.query_dataframe`.
- The test scripts `call_llm` (via the `scripted` fixture) with a decision sequence that walks the mandated order, then asserts the spec's measurable DoD.

- [ ] **Step 1: Write the scenario test**

```python
# tests/test_scenario.py
import pandas as pd
from agent.loop import (start_run, resume_run, get_pending_proposal, get_report, _graph_state)
from agent import store
from tools.query import query_dataframe

VALIDEE = {"decision": "validee", "motif_refus": None, "contre_proposition": None}


def test_end_to_end_mvp(scripted):
    df = None
    from data.generate_synthetic import generate_synthetic
    df = generate_synthetic(1000)

    scripted([
        ("ProfileDatasetArgs", {}),
        ("FixColumnTypesArgs", {"colonnes": ["date_commande", "quantite", "prix_unitaire", "montant_total"],
                                "types_cibles": {"date_commande": "date", "quantite": "entier",
                                                 "prix_unitaire": "decimal", "montant_total": "decimal"},
                                "justification": "Profilage : dates en texte (2 formats), numériques à fiabiliser."}),
        ("HandleDuplicatesArgs", {"sous_ensemble_colonnes": None, "justification": "~2% de doublons au profilage."}),
        # this one gets refused then counter-proposed:
        ("HandleMissingValuesArgs", {"colonnes": ["categorie", "mode_paiement"], "strategie": "drop_rows",
                                     "justification": "6% de manquants."}),
        ("TreatOutliersArgs", {"colonnes": ["quantite"], "action": "remove_rows",
                               "justification": "Quantités négatives = violation de règle métier (profilage: min < 0).",
                               "bornes_valides": {"min": 0, "max": None}}),
        ("EngineerFeaturesArgs", {"specification": {"type": "formula", "source": ["quantite", "prix_unitaire"],
                                  "output": "montant_total_recalc", "params": {"expression": "quantite * prix_unitaire"}},
                                  "justification": "montant_total incohérent sur une partie des lignes."}),
        ("GenerateReportArgs", {}),
    ])

    start_run("scenario", df)

    # 1. fix_column_types — validate
    assert get_pending_proposal("scenario")["tool_name"] == "fix_column_types"
    resume_run("scenario", VALIDEE)
    # 2. handle_duplicates — validate
    assert get_pending_proposal("scenario")["tool_name"] == "handle_duplicates"
    resume_run("scenario", VALIDEE)
    # 3. handle_missing_values drop_rows — REFUSE with counter-proposal (mode)
    assert get_pending_proposal("scenario")["tool_name"] == "handle_missing_values"
    resume_run("scenario", {"decision": "refusee", "motif_refus": "drop_rows trop destructif",
                            "contre_proposition": {"tool_name": "handle_missing_values",
                                "arguments": {"colonnes": ["categorie", "mode_paiement"], "strategie": "mode",
                                              "justification": "Imputation par le mode, non destructif."}}})
    # 4. treat_outliers — validate
    assert get_pending_proposal("scenario")["tool_name"] == "treat_outliers"
    resume_run("scenario", VALIDEE)
    # 5. engineer_features — validate
    assert get_pending_proposal("scenario")["tool_name"] == "engineer_features"
    resume_run("scenario", VALIDEE)

    df_current = store.get_current("scenario")

    # ---- measurable DoD (spec §"Definition of done") ----
    assert df_current.duplicated().sum() == 0
    assert df_current["categorie"].isna().mean() < 0.06
    assert df_current["quantite"].min() >= 0
    assert str(df_current["date_commande"].dtype) == "datetime64[ns]"

    # report exists, lists actions, distinguishes read vs modification
    report = get_report("scenario")
    assert report is not None
    assert "Lecture (auto)" in report
    assert "Modification (validée)" in report
    assert "Modification (contre-proposition humaine)" in report

    # query_dataframe answers the 3 test questions
    d = df_current[df_current["quantite"] > 0]
    q1 = query_dataframe(d, "groupby_agg", {"colonne_groupby": "produit", "colonne_agg": "quantite",
        "fonction": "sum", "periode": {"colonne_date": "date_commande", "mois": 3}})
    assert q1["status"] == "ok" and q1["detail"]["top"][0] is not None
    q2 = query_dataframe(d, "filter_count", {"colonne": "montant_total", "operateur": ">", "valeur": 1000})
    assert q2["status"] == "ok"
    q3 = query_dataframe(d, "top_n", {"colonne": "montant_total", "n": 5, "ordre": "desc"})
    assert q3["status"] == "ok" and len(q3["detail"]["lignes"]) == 5


def test_guardrail_cuts_when_nothing_is_validated(scripted):
    from data.generate_synthetic import generate_synthetic
    scripted([("FixColumnTypesArgs", {"colonnes": ["quantite"], "types_cibles": {"quantite": "entier"},
                                      "justification": "x"})] * 40)
    start_run("scenario_guard", generate_synthetic(500))
    # never resume the interrupt -> the graph is paused; drive think manually is not possible,
    # so this variant asserts the pause happened and no report was produced without human input.
    assert get_pending_proposal("scenario_guard") is not None
    assert get_report("scenario_guard") is None


def test_out_of_band_question_keeps_pending_proposal(scripted):
    from data.generate_synthetic import generate_synthetic
    scripted([
        ("HandleDuplicatesArgs", {"sous_ensemble_colonnes": None, "justification": "doublons"}),
        ("GenerateReportArgs", {}),
    ])
    start_run("scenario_oob", generate_synthetic(500))
    assert get_pending_proposal("scenario_oob")["tool_name"] == "handle_duplicates"
    # a question arrives while the proposal is pending
    from agent.loop import submit_question
    submit_question("scenario_oob", "Quelle est la distribution de montant_total ?")
    # proposal still pending, unchanged
    assert get_pending_proposal("scenario_oob")["tool_name"] == "handle_duplicates"
```

Notes:
- `test_guardrail_cuts_when_nothing_is_validated` is weaker than the spec's phrasing because a paused `interrupt` genuinely cannot advance without a human resume. If Task 13 instead implements the guardrail so that a configurable "auto-refuse after K unanswered proposals" path exists, strengthen this test. **Flag to Olivier:** the spec says "le garde-fou d'itérations coupe l'agent si on ne valide jamais rien" — but with a true `interrupt`, "never validating" means "never resuming", and a paused graph consumes no tokens (good) but also never reaches the guardrail. Clarify intended behaviour: (a) current — pause forever, no cost; (b) each resume with `refusee` counts toward the cap and eventually forces the report (this is already true — the guardrail fires after 15 `think` passes across refuse/revise cycles). Option (b) is what `test_refusal_*` + iteration counting already deliver; this test documents that "no resume at all" just stays paused.

- [ ] **Step 2: Run the scenario test**

Run: `uv run pytest tests/test_scenario.py -v`
Expected: PASS. Debug the loop/tools until all DoD assertions hold.

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest -v`
Expected: all green.

- [ ] **Step 4: Write `README.md`**

Include: what AutoPrep Agent is, the Console-account note (API billing separate from Claude Pro), setup (`uv sync`, copy `.env.example` → `.env`, add key), run (`uv run streamlit run app.py`), regenerate data, run tests, and the verbatim roadmap line:

> LLM actuel : Claude via API ; le point d'abstraction unique `call_llm()` permet de basculer sur un modèle local (Ollama) pour la souveraineté des données — roadmap SCIO.

Also document the two known limitations from the spec: `messages` growth over long sessions, and the `treat_outliers` `bornes_valides` provisional extension (Deviations §5).

- [ ] **Step 5: Commit**

```bash
git add tests/test_scenario.py README.md
git commit -m "test: end-to-end MVP scenario (definition of done) + README"
```

---

## Self-Review

**1. Spec coverage:**

| Spec element | Task |
|---|---|
| Besoin principal — agent decides, not pipeline | 13 (LLM-driven routing, order is prompt-only) |
| HITL validation before every write | 13 (`human_gate` + `interrupt`) |
| Out-of-band questions don't disturb pending proposal | 13, 15 (`submit_question`, `test_out_of_band_*`) |
| Iteration guardrail N=15 on `think` only | 1 (`config`), 13 (`think` increment + `force_report`) |
| Steering / counter-proposal via same schema | 13 (`human_gate` counter-proposal → `validate_args`) |
| Refusal carries `motif_refus` | 13, 15 |
| System prompt (verbatim) | 12 |
| Tool contracts — 5 read, 5 write, envelope | 3–10 |
| `query_dataframe` parameterised ops + `top_n` cap 20 + `periode` | 9 |
| `TOOL_REGISTRY` hard-coded labels/params, excludes `justification` | 3 |
| `fix_column_types` tolerant multi-format dates → NaT | 5 |
| Negative `quantite` = business rule via profile min/max, treated in step 5 | 4 (min/max), 7 (`bornes_valides` — flagged) |
| LangGraph state schema (all fields) | 11 |
| Dataframes out of state/`messages`, module dict by `thread_id` | 11 (`store.py`) |
| Proposition / réponse humaine format | 13 (`resume_run` payload), 14 (UI) |
| Synthetic generator — seeds, dates as raw str, all anomalies | 2 |
| Report format — fixed sections, read/write distinction, writes `rapport.md` | 10 |
| Error handling — tool errors as `status:error`, API retry, invalid upload | 3/5/6/7/8/9 (error envelopes), 12 (retry), 14 (upload) |
| French encoding/separator on upload | 14 |
| Project tree | 1 + all |
| DoD measurable assertions | 15 |
| `.env` in `.gitignore` before any `.env` | 1 |

No uncovered spec requirements. Gaps deliberately flagged: Deviations §3–§7.

**2. Placeholder scan:** `_guess_col` in Task 13 is a real stub (labelled, with a TODO and covered by Task 15 driving queries directly) — acceptable for MVP, flagged. No `TBD`/`implement later`/"add error handling" placeholders; every code step has runnable code.

**3. Type consistency:** envelope keys (`status/summary/metrics/detail`) consistent across Tasks 3–10. `actions_log` entry shape defined in Task 10 Interfaces and produced identically in Task 13 `act`. `dispatch` returns `(df|None, envelope)` in Task 3, consumed that way in Task 13. `MODEL_NAME_TO_TOOL` defined in Task 3, used in Task 13. `TOOL_TO_CHECKLIST` defined in Task 11, used in Task 13. `resume_run` payload shape (`decision/motif_refus/contre_proposition`) consistent between Tasks 13, 14, 15. `bornes_valides` shape (`{"min","max"}`) consistent Tasks 3, 7, 15.

---

## Execution Handoff

Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration. REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
2. **Inline Execution** — execute tasks in this session with checkpoints. REQUIRED SUB-SKILL: superpowers:executing-plans.

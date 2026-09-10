# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

The MVP is **implemented** per `autoprep-agent-architecture.md`, with a root-level layout: `agent/` (the LangGraph loop, state schema, system prompt, `llm_client`), `tools/` (the 10 tools plus `registry.py`), `data/generate_synthetic.py` (seeded Faker generator), `app.py` (Streamlit UI), and `tests/` (51 tests, including the end-to-end `tests/test_scenario.py` that encodes the definition of done). The architecture doc remains the source of truth for design decisions — read it before changing any agent/tool behavior, and prefer it over this summary. Run the suite with `uv run pytest`.

## Working method (see `methode-olivier-architecte.md`)

- **Olivier is the architect**: he decides and validates all architecture. Claude Code's job is to **implement what is documented**, not to redesign it.
- If something in `autoprep-agent-architecture.md` is ambiguous or looks inconsistent, flag it explicitly rather than silently picking an interpretation — Olivier's stated preference is to catch inconsistencies rather than have them papered over.
- When a design element is added or changed, verify it propagates everywhere it needs to (tool lists, registries, prompt text, state schema) — the architecture doc itself models this discipline (see its "Vérification de cohérence" notes).

## Commands

Dependency management is via `uv` (`pyproject.toml` + `uv.lock`). Python `>=3.14` (see `.python-version`).

```bash
uv sync                                      # install/sync dependencies into .venv
uv run streamlit run app.py                 # run the Streamlit UI (primary entry point)
uv run python -m data.generate_synthetic    # regenerate the synthetic dataset
uv run pytest                                # run the test suite
```

## Architecture (target design, per `autoprep-agent-architecture.md`)

**Core idea**: AutoPrep Agent is a LangGraph-based ReAct agent (not a fixed pipeline) that cleans/transforms an uploaded PME (SME) sales dataset and produces a valorization report. It observes real dataset state each cycle and decides its own next tool call, rather than following a hardcoded sequence.

**Non-negotiable constraint — human-in-the-loop**: any tool that *modifies* data (typing, duplicates, missing values, outliers, feature engineering — the 5 tools in `WRITE_TOOLS`) must pause for human validation (quoi/comment/pourquoi) before executing. Read-only tools (profiling, stats, charts, querying) execute freely.

**Planned file tree** (none of this exists yet):
```
app.py                     # Streamlit UI + thread_id orchestration
agent/loop.py              # LangGraph graph: think / act / interrupt nodes
agent/state.py             # TypedDict state schema
agent/prompts.py           # system prompt
agent/llm_client.py        # call_llm() — single abstraction point for the LLM provider
tools/registry.py          # TOOL_REGISTRY + WRITE_TOOLS classification + pydantic schemas
tools/profiling.py, typing_.py, cleaning.py, outliers.py, features.py, reporting.py, charts.py, query.py
data/generate_synthetic.py # Faker-based synthetic PME dataset generator (seeded)
tests/test_scenario.py     # end-to-end scenario, MVP definition of done
config.py                  # max_iterations, model name, thresholds (non-secret config)
```

**Key architectural decisions to preserve when implementing:**

- **LLM abstraction**: all LLM calls go through a single `call_llm()` in `agent/llm_client.py`. Current provider is the Claude API (Console account, not the user's Claude Pro subscription — these are billed separately); local Ollama models are a documented future roadmap item routed through this same function, not built now.
- **State lives in LangGraph, not `st.session_state`**: Streamlit only keeps `thread_id` and the compiled graph instance. Everything else (messages, iteration_count, checklist, pending_proposal, actions_log, etc. — full schema in the architecture doc) is LangGraph state, read/written only via the graph.
- **Dataframes never enter LangGraph state or `messages`**: `df_original`/`df_current` live in a module-level dict keyed by `thread_id` (checkpointing the state would otherwise serialize the whole dataframe every node). Tool results passed to the LLM carry only `summary` + `metrics`, never raw data.
- **Common tool return envelope**: `{"status": "ok"|"error", "summary": str, "metrics": dict, "detail": ...}`. Modifying tools must always populate before/after `metrics`.
- **Iteration guardrail**: `max_iterations = 15`, incremented only on the `think` node (not during interrupt/pause), forcing `generate_report` when exceeded — this is a normal, visible graph path, not a silent failure.
- **Out-of-band questions**: a user question arriving while a proposal is pending is answered (via `query_dataframe`) without disturbing the pending proposal.
- **Mandated tool order** (enforced by system prompt only, not hardcoded — deliberate, so the agent still "decides"): profile → fix_column_types → handle_duplicates → handle_missing_values → detect/treat_outliers → engineer_features. Exit only via an explicit checklist, never a vague "done" judgment from the LLM.
- **Business-rule violations vs statistical outliers are different mechanisms**: e.g. negative quantities are caught via min/max in `profile_dataset`, not via `detect_outliers` (IQR/z-score won't flag a business-rule violation).
- **French SME file quirks**: uploaded CSV/Excel exports are often `latin-1`-encoded with `;` separators (French Excel uses `,` as decimal separator). Upload validation must auto-detect separator and fall back `utf-8` → `latin-1`.
- **Synthetic data generator**: must seed both `Faker.seed(42)` and `np.random.seed(42)` for reproducibility, and write `date_commande` as raw strings on CSV export (letting pandas parse dates at save time would erase the intentionally mixed-format-date anomaly the agent is meant to detect).

**Security gap to fix before the first real commit**: `.gitignore` does not yet exclude `.env`, but the architecture mandates the API key live only in `.env`/environment variables. Add `.env` to `.gitignore` before introducing any `.env` file.

For full detail (exact tool signatures/enums, the complete system prompt, the full LangGraph state field list, the report format, error-handling rules, and the measurable MVP definition-of-done assertions), read `autoprep-agent-architecture.md` directly rather than relying on this summary.

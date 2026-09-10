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

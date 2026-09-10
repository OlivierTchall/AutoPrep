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

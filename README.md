# AutoPrep Agent

Agent d'aide à la préparation de données pour PME, construit sur **LangGraph** avec
**human-in-the-loop**. On lui fournit un export de ventes (CSV/Excel de PME
française), il **profile le jeu de données, propose des corrections** (types,
doublons, valeurs manquantes, valeurs aberrantes, variables dérivées) et produit
un **rapport de valorisation** en Markdown.

C'est un **agent ReAct**, pas un pipeline figé : à chaque cycle il observe l'état
réel du dataframe et décide lui-même de son prochain appel d'outil. L'ordre
métier recommandé (profilage → types → doublons → manquants → aberrants →
features) est induit par le *system prompt*, pas codé en dur.

## Human-in-the-loop

Les 5 outils qui **modifient** les données (`fix_column_types`,
`handle_duplicates`, `handle_missing_values`, `treat_outliers`,
`engineer_features`) passent par un `interrupt()` : l'agent expose le
**quoi / comment / pourquoi** et attend une décision humaine
(`validee` / `refusee` avec motif / contre-proposition via le même schéma
d'arguments). Les outils en lecture seule (profilage, stats, graphiques,
`query_dataframe`) s'exécutent librement.

Garde-fou : `max_iterations = 15`, incrémenté uniquement sur le nœud `think`.
Au-delà, l'agent génère le rapport avec l'état courant — c'est un chemin visible
du graphe, pas un échec silencieux.

## Compte API / facturation

Le fournisseur LLM actuel est l'**API Claude via un compte Console Anthropic**.
Cette facturation à l'usage est **distincte d'un abonnement Claude Pro ou de
Claude Code** — il faut une clé API Console dédiée, créditée séparément.

## Installation

```bash
uv sync                       # installe les dépendances dans .venv
cp .env.example .env          # puis éditer .env
```

Renseigner dans `.env` :

```
ANTHROPIC_API_KEY=sk-ant-...
# AUTOPREP_MODEL=claude-haiku-4-5-20251001   # optionnel
```

`.env` est ignoré par git ; la clé ne doit vivre que là ou dans l'environnement.

## Utilisation

```bash
uv run streamlit run app.py               # interface Streamlit (upload + validation)
uv run python -m data.generate_synthetic  # (re)génère data/ventes_pme.csv (seed fixe)
uv run pytest                             # lance la suite de tests
```

Le scénario end-to-end `tests/test_scenario.py` est la **définition de terminé**
du MVP : il pilote le vrai graphe et les vrais outils avec un LLM scripté et
vérifie les critères mesurables de la spec (0 doublon, < 6 % de manquants sur
`categorie`, `quantite` ≥ 0, `date_commande` en `datetime64[ns]`, rapport
distinguant lecture / modification validée / contre-proposition). La suite écrit
`rapport.md` à la racine (déjà gitignoré) — c'est attendu.

## Architecture

- `app.py` — UI Streamlit, ne conserve que `thread_id` + le graphe compilé.
- `agent/loop.py` — graphe LangGraph : `think` / `act` / `human_gate` / `report`.
- `agent/state.py` — schéma d'état (TypedDict). Tout l'état vit dans LangGraph.
- `agent/prompts.py` — system prompt.
- `agent/llm_client.py` — `call_llm()`, **point d'abstraction unique** du LLM.
- `agent/store.py` — les dataframes vivent dans un dict module-level indexé par
  `thread_id`, **jamais** dans l'état du graphe ni dans `messages`.
- `tools/` — outils + `registry.py` (`TOOL_REGISTRY`, `WRITE_TOOLS`, schémas
  pydantic, dispatcher). Enveloppe commune :
  `{"status", "summary", "metrics", "detail"}`.
- `data/generate_synthetic.py` — générateur Faker seedé (`Faker.seed(42)` +
  `np.random.seed(42)`).

## Roadmap

> LLM actuel : Claude via API ; le point d'abstraction unique `call_llm()` permet de basculer sur un modèle local (Ollama) pour la souveraineté des données — roadmap SCIO.

## Limitations connues

- **Croissance de `messages`** : l'historique des messages transmis au LLM
  grossit sans borne au fil d'une session longue (pas de troncature ni de
  résumé). Acceptable pour le MVP ; à traiter (fenêtrage / résumé) pour des
  sessions étendues.
- **`treat_outliers.bornes_valides`** : ce paramètre (plage métier valide, ex.
  `{"min": 0, "max": None}` pour cibler les quantités négatives) est une
  **extension approuvée** du contrat d'outil de `autoprep-agent-architecture.md`,
  ajoutée pour distinguer les **violations de règle métier** des vrais outliers
  statistiques (IQR / z-score). Voir la section Deviations §5 de la spec.

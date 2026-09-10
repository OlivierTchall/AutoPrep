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

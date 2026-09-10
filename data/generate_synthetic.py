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
    # Ensure date_commande is object dtype
    df["date_commande"] = df["date_commande"].astype(object)

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

    # 4. doublons ~2% (insérer des copies de lignes existantes)
    dup_count = int(0.02 * n)
    dup_src_idx = rng.choice(n, size=dup_count, replace=False)

    # Insert duplicates at random positions within the first n rows
    for i, src_idx in enumerate(sorted(dup_src_idx)):
        dup_row = df.iloc[src_idx].copy()
        # Random position to insert
        insert_pos = rng.integers(0, len(df) + 1)
        df = pd.concat([
            df.iloc[:insert_pos],
            pd.DataFrame([dup_row]),
            df.iloc[insert_pos:]
        ], ignore_index=True)

    # Keep only the first n rows
    df = df.iloc[:n].reset_index(drop=True)

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

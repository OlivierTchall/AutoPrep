# Méthode de travail d'Olivier — Architecte / Claude Code Codeur

Document vivant, rempli progressivement au fil des projets. Sert de base à un skill Claude Code une fois la méthode stabilisée.

## Répartition des rôles

- **Olivier — Architecte.** Conçoit l'architecture complète de la solution avec une vue transversale : comment chaque composant s'articule avec les autres avant qu'une seule ligne de code ne soit écrite. Décide, tranche, valide.
- **Claude (ici) — Capteur de méthode et documentaliste.** Ne décide pas de l'architecture à la place d'Olivier. Pose les questions qui aident à clarifier la vue transversale, documente au fur et à mesure ce qui est décidé, structure le cahier des charges final. Capture aussi les patterns récurrents de la façon de travailler d'Olivier pour en faire un skill réutilisable.
- **Claude Code — Codeur.** N'intervient qu'une fois l'architecture entièrement documentée. Implémente ce qui est écrit, ne redéfinit pas l'architecture.

## Principe central

Documentation complète de l'architecture **avant** tout passage au code. Le document remis à Claude Code doit être suffisamment précis pour qu'il n'ait pas à improviser de décisions d'architecture en cours de route.

## Processus observé (à affiner au fil des projets)

1. Olivier présente le projet / besoin.
2. Échange pour clarifier la vue transversale : composants, flux de données, interfaces entre composants, contraintes (sécurité, souveraineté des données, stack imposée, etc.).
3. Documentation formelle de l'architecture — format à stabiliser avec l'usage (schéma des composants, contrats d'interface, choix techniques justifiés, contraintes non négociables).
4. Remise du document à Claude Code pour implémentation.
5. (à valider) Revue du code produit par rapport à l'architecture documentée.

## Format de documentation — à définir avec l'usage

Points encore ouverts, à trancher au fil des projets :
- Schémas d'architecture : texte structuré, diagrammes (Mermaid), ou les deux ?
- Niveau de détail attendu sur les interfaces entre composants (signatures de fonctions ? contrats de données ? les deux ?)
- Les contraintes non fonctionnelles (sécurité, RGPD, souveraineté des données) sont systématiquement explicitées — confirmé sur le projet Agent IA (sandboxing de l'exécution de code piloté par LLM, architecture prête pour LLM local).

## Patterns observés dans la façon de travailler d'Olivier

- Quand Olivier découvre un domaine (ici l'IA agentique), il veut une documentation qui explique les **concepts et le fonctionnement**, pas seulement une spec technique à exécuter — et il demande explicitement des sources externes pour approfondir de son côté.
- Préférence marquée pour le contrôle : même dans un système qu'il conçoit comme autonome, il impose un garde-fou humain (validation avant les actions à risque, avec justification systématique — quoi / comment / pourquoi). À vérifier si ce réflexe (autonomie du système, mais jamais sans point de validation humaine explicite) se retrouve sur ses futurs projets — si oui, ça devient un principe par défaut du skill.
- Distingue explicitement les décisions d'architecture (siennes) des détails d'implémentation (délégués, une fois documentés).

- Revue technique très rigoureuse une fois qu'une proposition est posée : repère systématiquement les trous entre ce qui est déclaré fonctionner (ex. "l'agent détecte les outliers") et ce qui fonctionne réellement au niveau implémentation (ex. une quantité négative n'est pas un outlier statistique — aucune méthode IQR/z-score ne la verrait). Anticiper ce niveau de vérification en amont plutôt que d'attendre sa relecture.
- Exige des critères de validation **mesurables** plutôt que subjectifs — préfère des assertions testables (`df.duplicated().sum() == 0`) à des descriptions qualitatives ("l'agent a bien nettoyé les données").
- Préfère des schémas fermés (enums, objets typés) à du texte libre partout où un LLM ou un humain pourrait halluciner un paramètre invalide — vue comme la seule façon de laisser un codeur (humain ou Claude Code) ne rien avoir à deviner.
- Quand deux de ses propres propositions successives se chevauchent ou se contredisent légèrement (ex. deux formats de réponse humaine différents entre deux messages), il attend que Claude repère l'incohérence et la réconcilie explicitement plutôt que de la laisser passer.
- Rythme de travail sur un document d'architecture : proposition de Claude → corrections détaillées et justifiées d'Olivier, point par point → consolidation immédiate dans le fichier unique par Claude, avec republication à chaque tour. Un seul document vivant, jamais fragmenté entre plusieurs fichiers pour un même projet.
- Attentif aux détails spécifiques au marché français (encodage latin-1, séparateur point-virgule dans les exports Excel FR) — les intègre spontanément dans les specs techniques, pas seulement dans le produit final.
- Avant tout handoff à Claude Code, demande explicitement une passe de cohérence : quand un élément est ajouté en cours de conception (ex. un nouvel outil), il fait vérifier sa propagation partout où il doit apparaître dans le document, plutôt que de supposer que c'est fait. À anticiper systématiquement en fin de phase de conception, sans attendre qu'il le demande.

## Journal des projets traités

### Projet Agent IA Automatisation (AutoPrep Agent) — en cours
Démarré le 2026-09-09. Phase actuelle : architecture entièrement documentée (besoin, boucle de décision, prérequis techniques, prompt système, contrats des outils, schéma d'état LangGraph, générateur de données, format du rapport, gestion des erreurs, arborescence, definition of done). Reste : validation finale d'Olivier avant handoff à Claude Code. Voir `autoprep-agent-architecture.md` pour le détail complet.

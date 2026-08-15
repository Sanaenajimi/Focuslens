# 🎥 FocusLens — Analyse d'attention & de fatigue sur vidéo (fichier ou webcam)

> Pipeline temps réel robuste : détection de visage → 468 landmarks (MediaPipe) →
> métriques interprétables (EAR, MAR, pose 3D de tête) → analyse **temporelle**
> (fenêtres glissantes, hystérésis, calibration personnelle) → rapport complet.

## Ce que ça fait

Donne-lui **n'importe quelle vidéo** (mp4, avi, mov, webm) ou ta **webcam**, et il produit :

1. **Vidéo annotée** : landmarks, jauges EAR/attention, alertes somnolence/distraction incrustées
2. **Timeline CSV** : une ligne par frame (EAR, MAR, yaw/pitch/roll, score, événements)
3. **Rapport JSON** : stats agrégées (clignements/min, bâillements, % temps distrait, épisodes de somnolence…)
4. **Graphiques PNG** : score d'attention, EAR et pose dans le temps, avec les événements marqués

## Pourquoi c'est "robuste" (les vrais choix d'ingénierie)

| Problème réel | Solution implémentée |
|---|---|
| Chaque personne a des yeux différents (EAR de base ≠) | **Calibration automatique** sur les 3 premières secondes → seuils personnalisés |
| Une frame ratée ≠ un événement | **Hystérésis** : il faut N frames consécutives sous le seuil pour déclarer "yeux fermés", M au-dessus pour en sortir |
| Bruit frame-à-frame | **Lissage EMA** sur toutes les métriques + fenêtres glissantes pour les taux |
| Visage perdu (occlusion, sortie de champ) | Comptabilisé comme état "absent", n'invente jamais de valeurs, décroissance contrôlée du score |
| Vidéos verticales / rotées / fps exotiques | Lecture des métadonnées, fallback fps=30, redimensionnement borné |
| Clignement vs somnolence | Durée de fermeture : < 0.4 s = clignement ; > 1.0 s = micro-sommeil (PERCLOS-style) |

## Structure

```
focuslens/
├── src/
│   ├── config.py        # Tous les seuils/durées, documentés
│   ├── landmarks.py     # Wrapper MediaPipe FaceMesh (468 points)
│   ├── metrics.py       # EAR, MAR (bâillement), pose 3D (solvePnP)
│   ├── temporal.py      # EMA, hystérésis, calibration, événements
│   ├── analyzer.py      # Orchestrateur frame → FrameResult
│   ├── report.py        # Agrégation, JSON, CSV, graphiques
│   └── annotate.py      # Dessin de l'overlay sur la vidéo de sortie
├── cli.py               # python cli.py video.mp4  |  python cli.py --webcam
├── app/streamlit_app.py # Upload de vidéo + webcam, UI complète
├── requirements.txt
└── README.md
```

## Démarrage

```bash
python -m venv .venv  
.venv\Scripts\activate

pip install -r requirements.txt

# Analyser un fichier vidéo (sortie dans outputs/<nom_video>/)
python cli.py chemin/vers/ma_video.mp4

# Webcam en direct (q pour quitter)
python cli.py --webcam

# Interface web
streamlit run app/streamlit_app.py
```

## Datasets Kaggle pour aller plus loin

- **Drowsiness Detection** (yeux ouverts/fermés) → valider les seuils EAR sur données annotées
- **State Farm Distracted Driver** (10 classes) → ajouter un classifieur CNN de distraction
- **YawDD** (bâillements au volant) → valider le MAR

## Métriques implémentées

- **EAR** (Eye Aspect Ratio, Soukupová & Čech 2016) : ouverture des yeux, invariant à l'échelle
- **MAR** (Mouth Aspect Ratio) : détection de bâillement
- **Pose 3D** : yaw/pitch/roll par solvePnP sur 6 points anatomiques (regard hors caméra)
- **PERCLOS-lite** : % du temps yeux fermés sur fenêtre glissante (standard somnolence NHTSA)
- **Score d'attention composite /100** : présence + orientation + ouverture oculaire + rythme de clignement sain

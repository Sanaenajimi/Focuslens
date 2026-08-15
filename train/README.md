# 🧠 FocusLens · Module d'entraînement — TON propre CNN

> Ici tu n'utilises plus le modèle de Google. Tu **entraînes ton propre réseau de neurones**
> à reconnaître un œil ouvert vs fermé, à partir d'images. C'est le cœur "deep learning" du projet.

## Ce que tu vas entraîner

Un petit CNN (réseau de neurones convolutif) qui prend en entrée **une image d'œil (24×24 pixels, niveaux de gris)**
et sort **la probabilité que l'œil soit fermé**. Une fois entraîné, ce modèle remplace l'heuristique EAR
dans FocusLens : au lieu de calculer un ratio géométrique, on demande à un réseau *appris sur des données*.

```
Image d'œil (24×24) → [ TON CNN ] → probabilité "fermé" ∈ [0,1]
```

## Le dataset : MRL Eye Dataset (Kaggle)

Recommandé : **"MRL Eye Dataset"** ou **"Drowsiness_dataset"** sur Kaggle.
Structure attendue après téléchargement (à placer dans `train/data/`) :

```
train/data/
├── open/       ← des milliers d'images d'yeux OUVERTS
│   ├── img001.png
│   └── ...
└── closed/     ← des milliers d'images d'yeux FERMÉS
    ├── img001.png
    └── ...
```

Liens Kaggle (choisis-en un) :
- `kaggle datasets download -d prasadvpatil/mrl-dataset`
- `kaggle datasets download -d kutaykutlu/drowsiness-detection` (dossier Closed_Eyes / Open_Eyes)
- `kaggle datasets download -d hazemfahmy/openned-closed-eyes`

Expand-Archive -Path openned-closed-eyes.zip -DestinationPath dataset_raw -Force
Get-ChildItem dataset_raw -Recurse -Directory | Select-Object FullName  #pour visualiser le dossier

# Vide d'abord les README.txt que j'avais mis
Remove-Item train\data\open\README.txt -ErrorAction SilentlyContinue
Remove-Item train\data\closed\README.txt -ErrorAction SilentlyContinue

# Copie les images d'entraînement
Copy-Item "dataset_raw\TrainingSet\TrainingSet\Opened\*" train\data\open\ -Force
Copy-Item "dataset_raw\TrainingSet\TrainingSet\Closed\*" train\data\closed\ -Force

# Ajoute aussi le set d'amélioration (plus de données = meilleur modèle)
Copy-Item "dataset_raw\ImprovementSet\ImprovementSet\ImprovementSet\Opened\*" train\data\open\ -Force
Copy-Item "dataset_raw\ImprovementSet\ImprovementSet\ImprovementSet\Closed\*" train\data\closed\ -Force

Si les dossiers s'appellent `Closed_Eyes`/`Open_Eyes`, renomme-les `closed`/`open`,
ou passe-les en argument (voir `--open-dir` / `--closed-dir`).

## Étapes (tout est scripté)

```bash
pip install torch torchvision scikit-learn matplotlib pillow

# 1. ENTRAÎNER (sur CPU c'est OK, ~5-15 min selon la taille du dataset)
python train/train_eye_cnn.py --data train/data --epochs 12

#    → produit :
#      train/models/eye_cnn.pt         (les poids de TON modèle)
#      train/models/training_curves.png (courbes loss/accuracy)
#      train/models/confusion.png       (matrice de confusion)
#      train/models/metrics.json        (accuracy, AUROC, précision, rappel)


# 3. UTILISER ton modèle dans FocusLens
#    Dans src/config.py, mets : use_cnn_eye = True
#    FocusLens chargera automatiquement train/models/eye_cnn.pt
```

## Ce que tu apprends (et pourras expliquer en entretien)

- **Architecture CNN** : convolutions → pooling → couches denses, et POURQUOI chaque couche
- **La boucle d'entraînement** : forward → loss → backward → step (la mécanique de l'apprentissage)
- **Data augmentation** : pourquoi et comment on "invente" des variations pour généraliser
- **Régularisation** : dropout, early stopping, pour éviter le surapprentissage
- **Évaluation honnête** : split train/val/test, AUROC, matrice de confusion, pas juste l'accuracy
- **Déploiement** : sauvegarder, recharger, et brancher le modèle dans une app réelle

## Architecture du CNN (fichier `models/eye_cnn.py`)

```
Entrée : 1×24×24  (1 canal = niveaux de gris)
  Conv(1→16, 3×3) + ReLU + MaxPool  → 16×12×12   apprend bords/contours
  Conv(16→32,3×3) + ReLU + MaxPool  → 32×6×6      apprend formes (paupière, cils)
  Conv(32→64,3×3) + ReLU + MaxPool  → 64×3×3      apprend motifs complexes
  Flatten                           → 576
  Dropout(0.4)                                    régularisation
  Linear(576→64) + ReLU
  Linear(64→2)                      → logits [ouvert, fermé]
```

Léger (~120k paramètres), tourne en temps réel sur CPU — cohérent avec la philosophie embarquée de FocusLens.

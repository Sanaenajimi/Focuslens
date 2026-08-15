"""Architecture du CNN de détection d'œil ouvert/fermé.

Pédagogie : chaque couche est commentée avec CE QU'ELLE APPREND et POURQUOI.

Un CNN (Convolutional Neural Network) traite une image par couches successives :
- les premières couches apprennent des motifs SIMPLES (bords, contrastes) ;
- les couches profondes combinent ces motifs en concepts COMPLEXES
  (ici : "une paupière baissée", "des cils", "l'iris visible").

À la fin, deux couches "denses" transforment ces motifs en une décision : ouvert ou fermé.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EyeCNN(nn.Module):
    """Petit CNN (~120k paramètres) : image d'œil 24×24 niveaux de gris → 2 classes."""

    def __init__(self, dropout: float = 0.4):
        super().__init__()

        # --- BLOC CONVOLUTIF : l'extracteur de features visuelles ---
        # Chaque Conv2d applique des filtres qui glissent sur l'image.
        # MaxPool2d réduit la taille par 2 (garde l'info dominante, allège le calcul).
        self.features = nn.Sequential(
            # Couche 1 : 1 canal (gris) → 16 cartes de features. Apprend les BORDS.
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),      # stabilise et accélère l'entraînement
            nn.ReLU(inplace=True),   # non-linéarité : garde le positif, annule le négatif
            nn.MaxPool2d(2),         # 24×24 → 12×12

            # Couche 2 : 16 → 32. Combine les bords en FORMES (courbe de paupière...).
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),         # 12×12 → 6×6

            # Couche 3 : 32 → 64. Motifs COMPLEXES (œil ouvert = iris + blanc ; fermé = ligne).
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),         # 6×6 → 3×3
        )

        # --- TÊTE DE CLASSIFICATION : transforme les features en décision ---
        self.classifier = nn.Sequential(
            nn.Flatten(),                    # 64×3×3 = 576 valeurs mises à plat
            nn.Dropout(dropout),             # éteint 40% des neurones au hasard → anti-surapprentissage
            nn.Linear(64 * 3 * 3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),                # 2 logits : [score "ouvert", score "fermé"]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

    @torch.no_grad()
    def predict_closed_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne la probabilité 'fermé' (classe 1) via softmax."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)[:, 1]


# Constantes partagées avec l'entraînement ET l'inférence (cohérence garantie)
IMG_SIZE = 24
CLASS_NAMES = ["open", "closed"]  # index 0 = ouvert, index 1 = fermé

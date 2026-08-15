"""Configuration FocusLens — chaque seuil est documenté et justifié.

Règle senior : aucun nombre magique dans le code métier. Tout est ici,
modifiable, traçable, et sérialisable dans le rapport (reproductibilité).
"""
from dataclasses import dataclass, asdict


@dataclass
class FocusLensConfig:
    # --- Vidéo ---
    max_width: int = 960          # redimensionnement borné : précision suffisante, 3-4x plus rapide
    fallback_fps: float = 30.0    # certaines vidéos/webcams ne déclarent pas leur fps

    # --- Détection ---
    min_detection_conf: float = 0.5
    min_tracking_conf: float = 0.5

    # --- Détection œil : heuristique EAR (défaut) ou TON CNN entraîné ---
    use_cnn_eye: bool = False          # True → utilise train/models/eye_cnn.pt
    cnn_closed_threshold: float = 0.5  # proba au-dessus de laquelle l'œil est "fermé"

    # --- EAR (yeux) ---
    # Seuils RELATIFS à la baseline personnelle mesurée en calibration :
    # fermé si EAR < ratio_closed * baseline ; réouvert si EAR > ratio_open * baseline (hystérésis)
    calib_seconds: float = 5.0           # plus long = baseline plus fiable
    ear_ratio_closed: float = 0.60        # ABAISSÉ : lunettes compriment l'EAR de base
    ear_ratio_open: float = 0.75          # hystérésis suffisant
    ear_absolute_floor: float = 0.06     # ABAISSÉ : permet aux porteurs de lunettes d'être détectés
    blink_max_s: float = 0.55            # clignement normal jusqu'à 0.55 s
    microsleep_s: float = 1.20           # micro-sommeil dès 1.2 s (plus réactif)
    hysteresis_frames: int = 2           # réactivité augmentée

    # --- MAR (bâillement) ---
    mar_yawn: float = 0.60
    yawn_min_s: float = 0.8            # bouche ouverte < 0.8 s ≠ bâillement (parole)

    # --- Pose de tête ---
    yaw_away_deg: float = 25.0         # au-delà → regard hors caméra
    pitch_down_deg: float = 20.0       # tête baissée (téléphone ?)
    pose_away_min_s: float = 1.0       # hystérésis temporelle sur la distraction

    # --- Fenêtres temporelles ---
    ema_alpha: float = 0.25            # lissage des métriques (0=figé, 1=brut)
    perclos_window_s: float = 30.0     # fenêtre PERCLOS (standard : 30-60 s)
    perclos_severity_threshold: float = 0.15  # au-delà : le micro-sommeil est "critique"
                                                # (tendance de fond, pas un événement isolé)

    # --- Vélocité de fermeture/ouverture (dynamique du mouvement palpébral) ---
    # Un œil fatigué se ferme ET se rouvre plus LENTEMENT, indépendamment de la
    # durée totale de fermeture. On mesure la pente de l'EAR (dEAR/dt), normalisée
    # par la baseline personnelle → unité : "fractions de baseline par seconde".
    velocity_window_s: float = 0.35     # fenêtre glissante pour capter le pic de vitesse
    velocity_slow_threshold: float = 3.0  # en dessous : fermeture/ouverture jugée lente
    rate_window_s: float = 60.0        # fenêtre pour clignements/min

    # --- Score d'attention (pondérations, somme = 100) ---
    w_presence: float = 30.0
    w_facing: float = 30.0
    w_eyes: float = 25.0
    w_blink_rhythm: float = 15.0
    blink_healthy_min: int = 6         # 6-32 clignements/min = plage normale
    blink_healthy_max: int = 32
    score_floor_microsleep: float = 20.0

    def to_dict(self) -> dict:
        return asdict(self)
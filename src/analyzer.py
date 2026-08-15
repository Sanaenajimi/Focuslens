"""Analyzer : orchestre landmarks → métriques → temporel, frame par frame.

Une seule classe à connaître pour utiliser FocusLens en bibliothèque :

    fl = FocusLens()
    for frame, t in frames:
        result = fl.process(frame, t)
    fl.finalize(t)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

from .config import FocusLensConfig
from .landmarks import FaceLandmarker
from .metrics import ear_both, mouth_aspect_ratio, head_pose
from .temporal import TemporalAnalyzer


@dataclass
class FrameResult:
    t: float
    face_found: bool
    ear: float | None
    mar: float | None
    yaw: float | None
    pitch: float | None
    roll: float | None
    score: float | None
    eye_state: str
    perclos: float
    blinks_per_min: float
    microsleep: bool
    phase: str
    thr_closed: float | None = None      # seuil de fermeture appliqué (explicabilité)
    closed_duration: float = 0.0         # durée continue yeux fermés à cet instant (s)

    def to_row(self) -> dict:
        return asdict(self)

    def reason(self, cfg) -> str | None:
        """Justification textuelle de l'état courant, pour affichage live et rapport.
        Retourne None si rien de notable à expliquer (état neutre)."""
        if not self.face_found:
            return "Visage non détecté — aucune mesure possible."
        if self.microsleep:
            return (f"SOMNOLENCE : yeux fermés depuis {self.closed_duration:.1f}s "
                    f"(seuil d'alerte {cfg.microsleep_s:.1f}s) — EAR {self.ear:.2f} "
                    f"< seuil de fermeture {self.thr_closed:.2f}.")
        if self.eye_state == "closed" and self.closed_duration > 0:
            return (f"Yeux fermés depuis {self.closed_duration:.1f}s "
                    f"(EAR {self.ear:.2f} < seuil {self.thr_closed:.2f}) — "
                    f"sous le seuil d'alerte ({cfg.microsleep_s:.1f}s), probable clignement.")
        if self.yaw is not None and abs(self.yaw) > cfg.yaw_away_deg:
            return f"Regard hors route : orientation {self.yaw:+.0f}° (seuil ±{cfg.yaw_away_deg:.0f}°)."
        if self.perclos > cfg.perclos_severity_threshold:
            return f"Tendance de fond : {self.perclos*100:.0f}% du temps yeux fermés sur les 30 dernières secondes."
        return None


class FocusLens:
    def __init__(self, cfg: FocusLensConfig | None = None):
        self.cfg = cfg or FocusLensConfig()
        self.landmarker = FaceLandmarker(self.cfg.min_detection_conf, self.cfg.min_tracking_conf)
        self.temporal = TemporalAnalyzer(self.cfg)
        self.last_landmarks: np.ndarray | None = None

        # Chargement optionnel de TON CNN entraîné (remplace l'heuristique EAR).
        self.eye_cnn = None
        if self.cfg.use_cnn_eye:
            from .eye_cnn_infer import EyeStateCNN
            self.eye_cnn = EyeStateCNN()
            print("FocusLens : détection œil par CNN entraîné (eye_cnn.pt) ✅")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Redimensionnement borné : la précision des landmarks sature vers 960 px,
        au-delà on paie juste en latence. Robustesse fps/latence avant tout."""
        h, w = frame.shape[:2]
        if w > self.cfg.max_width:
            s = self.cfg.max_width / w
            frame = cv2.resize(frame, (self.cfg.max_width, int(h * s)),
                               interpolation=cv2.INTER_AREA)
        return frame

    def _cnn_pseudo_ear(self, frame: np.ndarray, pts: np.ndarray) -> float:
        """Le CNN sort une proba 'fermé'. On la convertit en pseudo-EAR pour que
        TOUTE la logique temporelle (calibration, hystérésis, durées) reste inchangée :
        proba élevée (œil fermé) → pseudo-EAR bas ; proba basse (ouvert) → pseudo-EAR haut.
        On ne remplace que la SOURCE du signal, pas le raisonnement autour."""
        from .landmarks import LEFT_EYE, RIGHT_EYE
        crops = [
            self.eye_cnn.eye_crop(frame, pts[LEFT_EYE]),
            self.eye_cnn.eye_crop(frame, pts[RIGHT_EYE]),
        ]
        p_closed = self.eye_cnn.proba_closed(crops)
        # mappe proba∈[0,1] vers pseudo-EAR∈[0.10, 0.34] (échelle EAR habituelle, inversée)
        return 0.34 - p_closed * 0.24

    def process(self, frame_bgr: np.ndarray, t: float) -> FrameResult:
        frame = self.preprocess(frame_bgr)
        pts = self.landmarker.process(frame)
        self.last_landmarks = pts

        if pts is None:
            state = self.temporal.update(t, False, None, None, None, None)
            return FrameResult(t, False, None, None, None, None, None,
                               state["score"], state["eye_state"], state["perclos"],
                               state["blinks_per_min"], state["microsleep"], state["phase"],
                               state.get("thr_closed"), state.get("closed_duration", 0.0))

        # Signal œil : CNN entraîné si activé, sinon heuristique EAR géométrique.
        if self.eye_cnn is not None:
            ear = self._cnn_pseudo_ear(frame, pts)
        else:
            ear = ear_both(pts)
        mar = mouth_aspect_ratio(pts)
        yaw, pitch, roll = head_pose(pts, frame.shape)
        state = self.temporal.update(t, True, ear, mar, yaw, pitch)
        return FrameResult(t, True, round(ear, 4), round(mar, 4),
                           round(yaw, 1), round(pitch, 1), round(roll, 1),
                           state["score"], state["eye_state"], state["perclos"],
                           state["blinks_per_min"], state["microsleep"], state["phase"],
                           state.get("thr_closed"), state.get("closed_duration", 0.0))

    def finalize(self, t_end: float) -> None:
        self.temporal.finalize(t_end)
        self.landmarker.close()
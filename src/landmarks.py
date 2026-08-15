"""Wrapper landmarks visage — DOUBLE BACKEND MediaPipe.

Les versions récentes de MediaPipe (Python 3.12/3.13, Windows notamment) ont
supprimé l'ancienne API `mp.solutions`. Ce module gère les deux mondes :

1. API legacy `mp.solutions.face_mesh` si disponible (rapide, zéro téléchargement)
2. Sinon, API moderne **Tasks** (`mediapipe.tasks.python.vision.FaceLandmarker`) :
   le modèle officiel (~4 Mo) est téléchargé automatiquement UNE fois dans
   ~/.focuslens/ puis réutilisé.

L'interface publique est identique dans les deux cas :
    FaceLandmarker(...).process(frame_bgr) -> np.ndarray (N, 3) en pixels | None
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

try:
    import mediapipe as mp
except ImportError as e:
    raise ImportError("MediaPipe manquant : pip install mediapipe") from e

# Indices (topologie FaceMesh — identiques dans les deux APIs)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = {"left": 61, "right": 291, "top": 13, "bottom": 14}
POSE_IDS = [1, 152, 33, 263, 61, 291]

_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task")
_MODEL_PATH = Path.home() / ".focuslens" / "face_landmarker.task"


def _ensure_model() -> Path:
    if not _MODEL_PATH.exists():
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print("FocusLens : téléchargement du modèle FaceLandmarker (~4 Mo, une seule fois)…")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


class _LegacyBackend:
    """Ancienne API mp.solutions.face_mesh (MediaPipe ≤ 0.10.x classiques)."""

    def __init__(self, det_conf: float, trk_conf: float):
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=det_conf, min_tracking_confidence=trk_conf)

    def process(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        h, w = frame_bgr.shape[:2]
        res = self._mesh.process(frame_bgr[:, :, ::-1])
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        return np.array([[p.x * w, p.y * h, p.z * w] for p in lm], dtype=np.float32)

    def close(self) -> None:
        self._mesh.close()


class _TasksBackend:
    """API moderne mediapipe.tasks (MediaPipe récents, Python 3.12+)."""

    def __init__(self, det_conf: float, trk_conf: float):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_ensure_model())),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=det_conf,
            min_face_presence_confidence=det_conf,
            min_tracking_confidence=trk_conf)
        self._lm = vision.FaceLandmarker.create_from_options(options)
        self._ts_ms = 0  # l'API VIDEO exige des timestamps strictement croissants

    def process(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._ts_ms += 33
        res = self._lm.detect_for_video(mp_img, self._ts_ms)
        if not res.face_landmarks:
            return None
        lm = res.face_landmarks[0]
        return np.array([[p.x * w, p.y * h, p.z * w] for p in lm], dtype=np.float32)

    def close(self) -> None:
        self._lm.close()


class FaceLandmarker:
    """Choisit automatiquement le backend disponible. Interface unique."""

    def __init__(self, min_detection_conf: float = 0.5, min_tracking_conf: float = 0.5):
        self.backend_name = "legacy"
        try:
            if not hasattr(mp, "solutions"):
                raise AttributeError
            self._backend = _LegacyBackend(min_detection_conf, min_tracking_conf)
        except (AttributeError, Exception):
            self.backend_name = "tasks"
            self._backend = _TasksBackend(min_detection_conf, min_tracking_conf)

    def process(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        return self._backend.process(frame_bgr)

    def close(self) -> None:
        self._backend.close()

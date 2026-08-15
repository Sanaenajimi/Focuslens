"""Analyse temporelle : là où un projet junior devient senior.

Une frame ne veut rien dire. Un CLIGNEMENT est une trajectoire d'EAR dans le
temps ; une SOMNOLENCE est une statistique sur 30 secondes. Ce module gère :

- Calibration : baseline EAR personnelle sur les premières secondes
- Hystérésis : deux seuils + N frames consécutives → pas d'oscillation d'état
- Machine à états oculaire : OPEN → CLOSING → CLOSED, avec durées
- Événements datés : blink, microsleep, yawn, look_away, absent
- Fenêtres glissantes : clignements/min, PERCLOS
- Score composite lissé (EMA)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .config import FocusLensConfig


@dataclass
class Event:
    kind: str          # blink | microsleep | yawn | look_away | absent
    t_start: float
    t_end: float
    severity: str = "info"   # info | modere | critique — voir _classify_severity
    closing_velocity: float | None = None   # pic de vitesse de fermeture (baseline/s, négatif)
    opening_velocity: float | None = None   # pic de vitesse de réouverture (baseline/s, positif)
    perclos_before: float | None = None     # PERCLOS mesuré juste avant l'événement (0-1)

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


class EMA:
    """Lissage exponentiel : new = α·x + (1−α)·old. Simple, causal, O(1)."""
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value: float | None = None

    def update(self, x: float) -> float:
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value


class TemporalAnalyzer:
    def __init__(self, cfg: FocusLensConfig):
        self.cfg = cfg
        # calibration
        self._calib_ears: list[float] = []
        self.baseline_ear: float | None = None
        # état yeux
        self._eye_state = "open"           # open | closed
        self._below_count = 0
        self._above_count = 0
        self._closed_since: float | None = None
        # état bouche / pose / présence
        self._mouth_open_since: float | None = None
        self._away_since: float | None = None
        self._absent_since: float | None = None
        # historique
        self.events: list[Event] = []
        self._blink_times: deque[float] = deque()
        self._perclos_buf: deque[tuple[float, bool]] = deque()   # (t, yeux fermés ?)
        # vélocité : fenêtre glissante des vitesses instantanées récentes (t, vel_normalisée)
        self._vel_buf: deque[tuple[float, float]] = deque()
        self._prev_ear: float | None = None
        self._prev_t: float | None = None
        self._closing_velocity: float | None = None
        self.score_ema = EMA(cfg.ema_alpha)
        self.ear_ema = EMA(cfg.ema_alpha)

    # ---------- calibration ----------
    def _calibrating(self, t: float) -> bool:
        return t < self.cfg.calib_seconds

    def _finish_calibration(self) -> None:
        if self._calib_ears:
            s = sorted(self._calib_ears)
            # 80e percentile : capture l'EAR "yeux grands ouverts" plutôt que
            # la médiane qui est tirée vers le bas par les clignements pendant
            # la calibration. Critique pour les porteurs de lunettes dont l'EAR
            # de base est naturellement plus bas (~0.10-0.18 au lieu de ~0.28).
            idx = int(0.80 * len(s))
            raw = s[min(idx, len(s) - 1)]
            self.baseline_ear = max(0.07, min(0.42, raw))
        else:
            self.baseline_ear = 0.25

    @property
    def thr_closed(self) -> float:
        base = self.baseline_ear or 0.25
        # ratio_closed = 0.60 : l'œil est "fermé" quand l'EAR tombe
        # à 60% de sa valeur de base. Avec EAR_base=0.12 (lunettes) :
        # thr_closed = max(0.06, 0.12×0.60) = max(0.06, 0.072) = 0.072
        # Quand l'œil se ferme vraiment → EAR chute vers 0.04-0.06 → détecté.
        return max(self.cfg.ear_absolute_floor, base * self.cfg.ear_ratio_closed)

    @property
    def thr_open(self) -> float:
        base = self.baseline_ear or 0.25
        return base * self.cfg.ear_ratio_open

    def _update_velocity(self, t: float, ear_s: float) -> None:
        """Calcule la vitesse instantanée de variation de l'EAR (dEAR/dt),
        normalisée par la baseline personnelle, et l'ajoute à la fenêtre
        glissante utilisée pour capter les pics de fermeture/ouverture."""
        if self._prev_ear is not None and self._prev_t is not None and t > self._prev_t:
            base = self.baseline_ear or 0.25
            raw_vel = (ear_s - self._prev_ear) / (t - self._prev_t)
            self._vel_buf.append((t, raw_vel / base))
        self._prev_ear, self._prev_t = ear_s, t
        while self._vel_buf and t - self._vel_buf[0][0] > self.cfg.velocity_window_s:
            self._vel_buf.popleft()

    def _peak_velocity(self, take_min: bool) -> float | None:
        """Pic de vitesse dans la fenêtre récente : minimum (fermeture, EAR chute,
        valeur négative) ou maximum (réouverture, EAR remonte, valeur positive)."""
        if not self._vel_buf:
            return None
        vals = [v for _, v in self._vel_buf]
        return min(vals) if take_min else max(vals)

    def _perclos_before(self, closure_start: float | None) -> float:
        """PERCLOS calculé sur la fenêtre glissante, en excluant toute frame
        appartenant à la fermeture en cours (t >= closure_start). C'est la
        tendance de fond QUI PRÉCÉDAIT cet événement — pas polluée par lui."""
        if closure_start is None or not self._perclos_buf:
            return 0.0
        prior = [c for tt, c in self._perclos_buf if tt < closure_start]
        if not prior:
            return 0.0
        return sum(prior) / len(prior)

    def _classify_severity(self, kind: str, duration: float, perclos_before: float,
                           closing_velocity: float | None, opening_velocity: float | None) -> str:
        """Qualifie un événement 'long_blink'/'microsleep' selon DEUX signaux
        indépendants, chacun suffisant à lui seul pour déclencher 'critique' :

        1. PERCLOS avant l'événement — la tendance de fond sur 30 s (le TEMPS).
        2. La vélocité de fermeture/ouverture — la dynamique du mouvement (la
           VITESSE). Un œil fatigué se ferme ET se rouvre plus lentement,
           indépendamment de la durée totale de fermeture : c'est le signal
           qu'un simple seuil de durée ne capture pas (cf. clignement volontaire
           long mais rapide, vs paupière qui "tombe" lentement).

        - un blink normal → INFO (jamais alarmant)
        - PERCLOS élevé OU dynamique lente → CRITIQUE (fatigue confirmée,
          par le temps ou par le mouvement — pas un accident isolé)
        - sinon → MODÉRÉ (événement isolé, rapide, pas de tendance de fond)
        """
        if kind == "blink":
            return "info"
        if perclos_before >= self.cfg.perclos_severity_threshold:
            return "critique"
        slow_close = closing_velocity is not None and abs(closing_velocity) < self.cfg.velocity_slow_threshold
        slow_open = opening_velocity is not None and abs(opening_velocity) < self.cfg.velocity_slow_threshold
        if slow_close and slow_open:
            return "critique"
        return "modere"

    # ---------- update par frame ----------
    def update(self, t: float, face_found: bool, ear: float | None,
               mar: float | None, yaw: float | None, pitch: float | None) -> dict:
        cfg = self.cfg

        # -- calibration en cours --
        if self._calibrating(t):
            if ear is not None:
                self._calib_ears.append(ear)
            return {"phase": "calibration", "score": None, "eye_state": "open",
                    "perclos": 0.0, "blinks_per_min": 0.0,
                    "thr_closed": None, "microsleep": False}
        if self.baseline_ear is None:
            self._finish_calibration()

        # -- présence --
        if not face_found:
            if self._absent_since is None:
                self._absent_since = t
        else:
            if self._absent_since is not None:
                self.events.append(Event("absent", self._absent_since, t))
                self._absent_since = None

        microsleep_now = False
        eyes_closed = False

        if face_found and ear is not None:
            ear_s = self.ear_ema.update(ear)
            self._update_velocity(t, ear_s)

            # -- machine à états yeux avec hystérésis --
            if self._eye_state == "open":
                self._below_count = self._below_count + 1 if ear_s < self.thr_closed else 0
                if self._below_count >= cfg.hysteresis_frames:
                    # Transition vers "fermé" : on capture le pic de vitesse de
                    # FERMETURE observé juste avant cet instant (fenêtre récente).
                    self._eye_state, self._closed_since = "closed", t
                    self._closing_velocity = self._peak_velocity(take_min=True)
                    self._below_count = 0
            else:  # closed
                eyes_closed = True
                dur = t - (self._closed_since or t)
                if dur >= cfg.microsleep_s:
                    microsleep_now = True
                self._above_count = self._above_count + 1 if ear_s > self.thr_open else 0
                if self._above_count >= cfg.hysteresis_frames:
                    kind = "blink" if dur <= cfg.blink_max_s else \
                           ("microsleep" if dur >= cfg.microsleep_s else "long_blink")
                    # PERCLOS calculé sur l'historique STRICTEMENT antérieur à cette
                    # fermeture (on exclut les frames de l'événement en cours, sinon
                    # un micro-sommeil isolé "pollue" sa propre mesure de tendance).
                    perclos_before = self._perclos_before(self._closed_since)
                    # Vitesse de RÉOUVERTURE : pic observé dans la fenêtre récente,
                    # au moment où l'œil vient de se rouvrir.
                    opening_velocity = self._peak_velocity(take_min=False)
                    severity = self._classify_severity(
                        kind, dur, perclos_before, self._closing_velocity, opening_velocity)
                    self.events.append(Event(kind, self._closed_since, t, severity,
                                             self._closing_velocity, opening_velocity, perclos_before))
                    if kind == "blink":
                        self._blink_times.append(t)
                    self._eye_state, self._closed_since = "open", None
                    self._above_count = 0
                    self._closing_velocity = None

            # -- bâillement (durée minimale : exclut la parole) --
            if mar is not None and mar > cfg.mar_yawn:
                if self._mouth_open_since is None:
                    self._mouth_open_since = t
            else:
                if self._mouth_open_since is not None:
                    if t - self._mouth_open_since >= cfg.yawn_min_s:
                        self.events.append(Event("yawn", self._mouth_open_since, t))
                    self._mouth_open_since = None

            # -- distraction pose (hystérésis temporelle) --
            away = yaw is not None and (abs(yaw) > cfg.yaw_away_deg or
                                        (pitch is not None and pitch < -cfg.pitch_down_deg))
            if away:
                if self._away_since is None:
                    self._away_since = t
            else:
                if self._away_since is not None:
                    if t - self._away_since >= cfg.pose_away_min_s:
                        self.events.append(Event("look_away", self._away_since, t))
                    self._away_since = None

        # -- fenêtres glissantes --
        self._blink_times and None
        while self._blink_times and t - self._blink_times[0] > cfg.rate_window_s:
            self._blink_times.popleft()
        self._perclos_buf.append((t, eyes_closed or not face_found))
        while self._perclos_buf and t - self._perclos_buf[0][0] > cfg.perclos_window_s:
            self._perclos_buf.popleft()
        perclos = sum(c for _, c in self._perclos_buf) / max(1, len(self._perclos_buf))
        elapsed_min = min(cfg.rate_window_s, max(t, 1e-6)) / 60.0
        bpm = len(self._blink_times) / elapsed_min

        # -- score composite --
        score = 0.0
        if face_found:
            score += cfg.w_presence
            if yaw is not None:
                score += cfg.w_facing * max(0.0, 1 - abs(yaw) / (2 * cfg.yaw_away_deg))
            if ear is not None and self.baseline_ear:
                score += cfg.w_eyes * min(1.0, max(0.0, self.ear_ema.value / self.baseline_ear))
            score += cfg.w_blink_rhythm if cfg.blink_healthy_min <= bpm <= cfg.blink_healthy_max \
                     else cfg.w_blink_rhythm * 0.5
        if microsleep_now:
            score = min(score, cfg.score_floor_microsleep)
        score_s = self.score_ema.update(score)

        return {"phase": "analysis", "score": round(score_s, 1),
                "eye_state": self._eye_state, "perclos": round(perclos, 3),
                "blinks_per_min": round(bpm, 1), "thr_closed": round(self.thr_closed, 3),
                "microsleep": microsleep_now,
                "closed_duration": round(t - self._closed_since, 2) if self._closed_since else 0.0}

    def finalize(self, t_end: float) -> None:
        """Clôt les épisodes encore ouverts en fin de vidéo."""
        if self._closed_since is not None:
            dur = t_end - self._closed_since
            kind = "microsleep" if dur >= self.cfg.microsleep_s else "blink"
            self.events.append(Event(kind, self._closed_since, t_end))
        if self._absent_since is not None:
            self.events.append(Event("absent", self._absent_since, t_end))
        if self._away_since is not None and t_end - self._away_since >= self.cfg.pose_away_min_s:
            self.events.append(Event("look_away", self._away_since, t_end))
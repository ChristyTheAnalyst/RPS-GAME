#!/usr/bin/env python3

import argparse
import json
import socket
import time
from collections import deque, Counter

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as e:
    raise SystemExit("mediapipe not installed. Run: pip install mediapipe")

# ------------------------- Utilities -------------------------

from datetime import datetime

def now_ts():
    # return local system date/time, e.g. "2025-08-19 18:25:30"
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def hand_center_and_size(hand_landmarks, image_h: int, image_w: int):
    """Return center (cx, cy) and bbox height (box_h) in normalized [0..1] coords.
    Robust to finger pose by using tight bbox across all 21 landmarks.
    """
    lm = hand_landmarks.landmark
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5
    box_h = max(1e-6, max_y - min_y)
    return float(cx), float(cy), float(box_h)


class TCPClient:
    def __init__(self, host: str, port: int, connect_timeout=2.0):
        self.host = host
        self.port = port
        self.sock = None
        self.connect_timeout = connect_timeout
        self.ensure_connected()

    def ensure_connected(self):
        if self.sock is not None:
            return
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.connect_timeout)
        s.connect((self.host, self.port))
        s.settimeout(None)
        self.sock = s

    def send_json(self, obj: dict):
        line = json.dumps(obj, separators=(",", ":")) + "\r\n"
        try:
            self.ensure_connected()
            self.sock.sendall(line.encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError, OSError):
            # try one reconnect, then drop silently
            try:
                if self.sock:
                    self.sock.close()
            except Exception:
                pass
            self.sock = None
            try:
                self.ensure_connected()
                self.sock.sendall(line.encode("utf-8"))
            except Exception:
                pass


# ------------------------- Gesture Classifier -------------------------

class GestureClassifier:
    """
    Angle-based, orientation-robust classifier.
    - A finger is STRAIGHT if both (MCP–PIP–TIP) and (PIP–DIP–TIP) angles ≈ 180°.
    - A finger is CURLED if the PIP joint angle is much smaller (e.g., < 150°).
    """

    def __init__(self,
                 straight_thresh_deg: float = 25.0,  # 180 - this => required straightness (e.g., 155°+)
                 curl_pip_thresh_deg: float = 150.0, # PIP angle below this => curled
                 ):
        self.straight_thresh_deg = straight_thresh_deg
        self.curl_pip_thresh_deg = curl_pip_thresh_deg

    @staticmethod
    def _angle_deg(a, b, c):
        """Angle ABC in degrees using 2D (x,y). Works regardless of hand pointing up/down."""
        import numpy as np, math
        v1 = np.array([a.x - b.x, a.y - b.y], dtype=float)
        v2 = np.array([c.x - b.x, c.y - b.y], dtype=float)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 180.0
        cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        return math.degrees(math.acos(cosang))

    def _is_straight(self, lm, mcp, pip, dip, tip) -> bool:
        need = 180.0 - self.straight_thresh_deg  # e.g., 155°
        ang_pip = self._angle_deg(lm[mcp], lm[pip], lm[tip])
        ang_dip = self._angle_deg(lm[pip], lm[dip], lm[tip])
        return (ang_pip >= need) and (ang_dip >= need)

    def _is_curled(self, lm, mcp, pip, dip, tip) -> bool:
        # Use PIP joint angle; small angle => bent.
        ang_pip = self._angle_deg(lm[mcp], lm[pip], lm[dip])
        return ang_pip < self.curl_pip_thresh_deg  # e.g., < 150° means curled

    def classify(self, hand_landmarks, image_h, image_w) -> str | None:
        lm = hand_landmarks.landmark

        # Indices per finger
        #  thumb: 1-2-3-4 (we ignore thumb for R/P/S)
        idx  = (5, 6, 7, 8)
        mid  = (9, 10, 11, 12)
        ring = (13, 14, 15, 16)
        pink = (17, 18, 19, 20)

        # Straight / Curled (angle-based, rotation-invariant)
        idx_st  = self._is_straight(lm, *idx)
        mid_st  = self._is_straight(lm, *mid)
        ring_st = self._is_straight(lm, *ring)
        pink_st = self._is_straight(lm, *pink)

        ring_curled  = self._is_curled(lm, *ring)
        pinky_curled = self._is_curled(lm, *pink)

        # ---- SCISSORS: index & middle straight, ring & pinky curled
        if idx_st and mid_st and ring_curled and pinky_curled:
            return "scissors"

        # ---- PAPER: all four non-thumb fingers straight and not curled
        if idx_st and mid_st and ring_st and pink_st and (not ring_curled) and (not pinky_curled):
            return "paper"

        # ---- ROCK: 0–1 straight (thumb ignored)
        straight_count = sum([idx_st, mid_st, ring_st, pink_st])
        if straight_count <= 1:
            return "rock"

        return None





# ------------------------- Shake Detector -------------------------

class ShakeDetector:
    """
    Counts FULL up–down cycles from the **hand center** (bbox center of all 21 landmarks).

    """

    def __init__(
        self,
        ema_alpha: float = 0.5,         # higher alpha for snappier tracking of fast motion
        vel_deadzone: float = 0.0008,   # ignore tiny velocity jitter
        rel_amp_frac: float = 0.12,     # min ΔY vs hand bbox height
        min_abs_amp: float = 0.015,     # absolute floor for ΔY (normalized)
        min_frames_between_extrema: int = 2,  # allow shorter gaps between peaks
        max_frames_between_extrema: int = 50,
        settle_frames_after_extrema: int = 4,  # quicker settle time after last shake
        target_cycles: int = 3,         # keep 3 shakes to trigger throw (tune as needed)
    ):
        self.ema_alpha = ema_alpha
        self.vel_deadzone = vel_deadzone
        self.rel_amp_frac = rel_amp_frac
        self.min_abs_amp = min_abs_amp
        self.min_frames_between_extrema = min_frames_between_extrema
        self.max_frames_between_extrema = max_frames_between_extrema
        self.settle_frames_after_extrema = settle_frames_after_extrema
        self.target_cycles = target_cycles

        self.y_s_prev = None
        self.last_extrema_y = None
        self.last_extrema_type = None   # 'max' or 'min'
        self.last_vel_sign = 0
        self.frames_since_extrema = 9999
        self.half_extrema = 0
        self.shakes = 0
        self.last_motion_time = time.time()

        self._dbg = {}

    def reset(self):
        self.y_s_prev = None
        self.last_extrema_y = None
        self.last_extrema_type = None
        self.last_vel_sign = 0
        self.frames_since_extrema = 9999
        self.half_extrema = 0
        self.shakes = 0
        self.last_motion_time = time.time()
        self._dbg = {}

    def _ema(self, y: float) -> float:
        if self.y_s_prev is None:
            self.y_s_prev = y
        else:
            a = self.ema_alpha
            self.y_s_prev = a * y + (1 - a) * self.y_s_prev
        return self.y_s_prev

    def update(self, center_y_norm: float, scale: float):
        """
        center_y_norm: bbox center Y in [0..1] (y grows downward)
        scale: hand bbox height in [0..1] (for adaptive amplitude)
        """
        y = float(center_y_norm)
        y_s_prev = self.y_s_prev if self.y_s_prev is not None else y
        y_s = self._ema(y)

        # Velocity of smoothed center (positive = moving down)
        dv = y_s - y_s_prev
        if abs(dv) < self.vel_deadzone:
            sign = 0
        else:
            sign = 1 if dv > 0 else -1

        if sign != 0:
            self.last_motion_time = time.time()

        self.frames_since_extrema += 1

        # Zero-crossing of velocity -> extremum around here
        extremum = (sign != 0 and self.last_vel_sign != 0 and sign != self.last_vel_sign)
        if extremum:
            # If we were going up (last_vel_sign < 0) and now reversed -> reached a MAX
            curr_type = 'max' if self.last_vel_sign < 0 else 'min'

            # Timing window
            good_timing = (self.min_frames_between_extrema <= self.frames_since_extrema <=
                           self.max_frames_between_extrema)

            # Adaptive amplitude threshold (relative to hand size, with absolute floor)
            dyn_thr = max(self.min_abs_amp, self.rel_amp_frac * max(scale, 1e-6))
            amp_ok = True if self.last_extrema_y is None else abs(y_s - self.last_extrema_y) >= dyn_thr

            if good_timing and amp_ok:
                if self.last_extrema_type is None or curr_type != self.last_extrema_type:
                    self.half_extrema += 1
                    self.shakes = self.half_extrema // 2  # two alternating extrema = 1 shake
                    self.last_extrema_type = curr_type
                    self.last_extrema_y = y_s
                    self.frames_since_extrema = 0
            else:
                # rebaseline but don't count
                self.last_extrema_type = curr_type
                self.last_extrema_y = y_s
                self.frames_since_extrema = 0

        if sign != 0:
            self.last_vel_sign = sign

        # debug (optional)
        self._dbg = {
            "y": round(y_s, 5),
            "dv": round(dv, 5),
            "scale": round(scale, 4),
            "dyn_thr": round(max(self.min_abs_amp, self.rel_amp_frac * max(scale, 1e-6)), 5),
            "frames_since_extrema": self.frames_since_extrema,
            "half_extrema": self.half_extrema,
            "shakes": self.shakes
        }

    def is_ready(self) -> bool:
        return (self.shakes >= self.target_cycles and
                self.frames_since_extrema >= self.settle_frames_after_extrema)

    def seconds_since_motion(self) -> float:
        return max(0.0, time.time() - (self.last_motion_time or time.time()))

    def debug_info(self) -> dict:
        return dict(self._dbg)


# ------------------------- Controller -------------------------

class RPSController:
    def __init__(self, arm_host: str, arm_port: int, camera_index=0, show_window=True):
        self.arm = TCPClient(arm_host, arm_port)
        self.show_window = show_window

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.detector = ShakeDetector()  # set target_cycles=1 here if you want a single shake to trigger
        self.classifier = GestureClassifier()

        self.state = "idle"  # idle | shaking | throw_prep | thrown
        self.last_state_change = now_ts()

    def set_state(self, s: str):
        if s != self.state:
            self.state = s
            self.last_state_change = now_ts()
            self.arm.send_json({"cmd": "state", "value": self.state, "ts": now_ts()})

    @staticmethod
    def winning_move(user_move: str) -> str:
        beats = {"rock": "paper", "paper": "scissors", "scissors": "rock"}
        return beats.get(user_move, "rock")

    def run(self):
        vote_window = deque(maxlen=7)  # ~200ms at 30fps
        prev_shakes = 0
        reset_time = None  # when not None, reset detector after this timestamp

        while True:
            ok, frame = self.cap.read()
            if not ok:
                break

            # deferred reset after completed handshake
            if reset_time is not None and time.time() >= reset_time:
                self.detector.reset()
                prev_shakes = 0
                reset_time = None
                self.set_state("idle")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)

            h, w = frame.shape[:2]
            user_move = None

            if res.multi_hand_landmarks:
                hand_landmarks = res.multi_hand_landmarks[0]
                # draw landmarks for debugging
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                cx, cy, box_h = hand_center_and_size(hand_landmarks, h, w)
                self.detector.update(cy, scale=box_h)

                # --- Send once when shaking begins (0 -> >0) ---
                if prev_shakes == 0 and self.detector.shakes > 0:
                    self.arm.send_json({
                        "cmd": "info",
                        "message": "Handshake started",
                        "ts": now_ts(),
                    })

                # Inactivity reset disabled; handshake count persists until manual reset

                # classify continuously (we'll only act on it on 3rd shake)
                guess = self.classifier.classify(hand_landmarks, h, w)
                if guess:
                    vote_window.append(guess)

            # State machine
            if reset_time is None:
                if self.detector.shakes > 0:
                    self.set_state("shaking")
                else:
                    self.set_state("idle")

            if self.detector.is_ready() and reset_time is None:
                self.set_state("throw_prep")

                # --- NEW: notify that the handshake sequence finished ---
                self.arm.send_json({
                    "cmd": "info",
                    "message": "Handshake finished",
                    "shakes": self.detector.shakes,
                    "ts": now_ts(),
                })

                # majority vote over last few frames
                if vote_window:
                    cnt = Counter(vote_window)
                    user_move = cnt.most_common(1)[0][0]
                    robot_move = self.winning_move(user_move)
                    payload = {
                        "cmd": "rps_throw",
                        "user": user_move,
                        "robot": robot_move,
                        "ts": now_ts(),
                    }
                    self.arm.send_json(payload)
                    self.set_state("thrown")
                    # brief freeze to show result, then reset after delay
                    vote_window.clear()
                    reset_time = time.time() + 1.5

            # HUD overlay
            hud = [
                f"State: {self.state}",
                f"Shakes: {self.detector.shakes} / {self.detector.target_cycles}",
                f"Gesture: {vote_window[-1] if vote_window else '-'}",
            ]
            for i, text in enumerate(hud):
                cv2.putText(frame, text, (10, 30 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            prev_shakes = self.detector.shakes

            if self.show_window:
                cv2.imshow("RPS Controller", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break
                elif key == ord(' '):
                    self.detector.reset()
                    vote_window.clear()
                    reset_time = None
                    self.arm.send_json({
                        "cmd": "info",
                        "message": "Handshake count reset",
                        "ts": now_ts(),
                    })
                    prev_shakes = 0
                    self.set_state("idle")

        self.cap.release()
        cv2.destroyAllWindows()


def parse_args():
    ap = argparse.ArgumentParser(description="Rock–Paper–Scissors controller → robotic arm via TCP")
    ap.add_argument("--arm-host", required=True, help="Robotic arm TCP server host/IP")
    ap.add_argument("--arm-port", type=int, default=5050, help="Robotic arm TCP server port")
    ap.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default 0)")
    ap.add_argument("--no-window", action="store_true", help="Disable preview window")
    return ap.parse_args()


def main():
    args = parse_args()
    ctrl = RPSController(args.arm_host, args.arm_port, args.camera, show_window=not args.no_window)
    ctrl.run()


if __name__ == "__main__":
    main()

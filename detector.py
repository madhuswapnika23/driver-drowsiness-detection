"""
Driver Drowsiness Detection — Level 1 Upgraded
Features:
  - EAR  : Eye Aspect Ratio (eye closure)
  - MAR  : Mouth Aspect Ratio (yawn detection)
  - PERCLOS : % eye closure over rolling 60-second window
  - Head pose : pitch angle to detect head nodding/drooping

Author: Madhu Swapnika G et al. | CSE303, SRM University-AP
"""

import cv2
import numpy as np
import time
import threading
import os
import sys
import argparse
import urllib.request
from collections import deque

# ── optional sound ─────────────────────────────────────────────────────────────
try:
    from playsound import playsound
    _SOUND = "playsound"
except ImportError:
    try:
        import winsound
        _SOUND = "winsound"
    except ImportError:
        _SOUND = "beep"

# ── MediaPipe Tasks API ────────────────────────────────────────────────────────
import mediapipe as mp
from mediapipe.tasks.python.vision import (
    FaceLandmarker, FaceLandmarkerOptions, RunningMode
)
from mediapipe.tasks.python.core.base_options import BaseOptions

# ══════════════════════════════════════════════════════════════════════════════
# LANDMARK INDICES
# ══════════════════════════════════════════════════════════════════════════════

# Eye landmarks (Soukupová & Čech P1-P6 order)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Mouth landmarks for MAR (top-lip, bottom-lip, left corner, right corner)
# Vertical pairs: (13,14), (312,317), (82,87)  Horizontal: (78,308)
MOUTH_MAR = [78, 82, 13, 312, 87, 317, 14, 308]
#             left  v1   v2   v3   v4   v5  v6  right

# Head pose model points (3D canonical face, mm)
MODEL_POINTS_3D = np.array([
    (0.0,    0.0,    0.0),      # Nose tip           (landmark 1)
    (0.0,  -330.0, -65.0),      # Chin               (landmark 152)
    (-225.0, 170.0,-135.0),     # Left eye corner    (landmark 33)
    (225.0,  170.0,-135.0),     # Right eye corner   (landmark 263)
    (-150.0,-150.0,-125.0),     # Left mouth corner  (landmark 61)
    (150.0, -150.0,-125.0),     # Right mouth corner (landmark 291)
], dtype=np.float64)

POSE_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

# ══════════════════════════════════════════════════════════════════════════════
# THRESHOLDS & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
EAR_THRESH        = 0.25    # Eye closed below this
EAR_CONSEC        = 20      # Consecutive frames → drowsy alert
MAR_THRESH        = 0.65    # Mouth open above this → yawn
MAR_CONSEC        = 15      # Consecutive frames → yawn alert
PERCLOS_WINDOW    = 60      # Rolling window in seconds
PERCLOS_THRESH    = 0.20    # 20 % eye closure → fatigue
PITCH_THRESH      = 20.0    # Head nod angle (degrees) → drowsy
PITCH_CONSEC      = 30      # Consecutive frames of nodding → alert
COOLDOWN_SEC      = 3       # Seconds between repeated beeps

MODEL_PATH  = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
SOUND_PATH  = os.path.join(os.path.dirname(__file__), "sounds", "alert.wav")
MODEL_URL   = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def download_model():
    if os.path.exists(MODEL_PATH):
        return
    print("[INFO] Downloading face_landmarker.task (~2 MB, first run only)…")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Model downloaded.")
    except Exception as e:
        sys.exit(f"[ERROR] Download failed: {e}\nGet it from:\n{MODEL_URL}")


def dist(a, b):
    return np.linalg.norm(np.subtract(a, b))


def to_px(lm, idx, W, H):
    """Convert normalized landmark to pixel coords."""
    return (lm[idx].x * W, lm[idx].y * H)


def compute_ear(lms, indices, W, H):
    """Eye Aspect Ratio — Soukupová & Čech 2016."""
    p = [to_px(lms, i, W, H) for i in indices]
    return (dist(p[1], p[5]) + dist(p[2], p[4])) / (2.0 * dist(p[0], p[3]))


def compute_mar(lms, W, H):
    """
    Mouth Aspect Ratio.
    MAR = (sum of 3 vertical mouth distances) / (2 * horizontal mouth distance)
    Using 6 points: left corner, right corner, and 3 vertical pairs.
    """
    left   = np.array(to_px(lms, 78,  W, H))
    right  = np.array(to_px(lms, 308, W, H))
    top1   = np.array(to_px(lms, 82,  W, H))
    bot1   = np.array(to_px(lms, 87,  W, H))
    top2   = np.array(to_px(lms, 13,  W, H))
    bot2   = np.array(to_px(lms, 14,  W, H))
    top3   = np.array(to_px(lms, 312, W, H))
    bot3   = np.array(to_px(lms, 317, W, H))

    vertical   = (np.linalg.norm(top1 - bot1) +
                  np.linalg.norm(top2 - bot2) +
                  np.linalg.norm(top3 - bot3))
    horizontal = np.linalg.norm(left - right)
    return vertical / (2.0 * horizontal + 1e-6)


def compute_pitch(lms, W, H, cam_matrix, dist_coeffs):
    """
    Returns pitch angle in degrees using solvePnP.
    Positive pitch = chin down (nodding / head drooping).
    """
    image_pts = np.array(
        [to_px(lms, i, W, H) for i in POSE_LANDMARK_IDS],
        dtype=np.float64
    )
    ok, rvec, _ = cv2.solvePnP(
        MODEL_POINTS_3D, image_pts, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    # Decompose rotation matrix to Euler angles
    sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(-rmat[2,0], sy))
    else:
        pitch = np.degrees(np.arctan2(-rmat[2,0], sy))
    return float(pitch)


def _beep():
    if _SOUND == "playsound" and os.path.exists(SOUND_PATH):
        try:
            playsound(SOUND_PATH, block=False)
            return
        except Exception:
            pass
    if _SOUND == "winsound":
        try:
            winsound.Beep(1000, 500)
            return
        except Exception:
            pass
    print("\a", end="", flush=True)


def alert(last_t_ref):
    """Play alert if cooldown has passed. Returns updated timestamp."""
    now = time.time()
    if now - last_t_ref[0] > COOLDOWN_SEC:
        last_t_ref[0] = now
        threading.Thread(target=_beep, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# HUD DRAWING
# ══════════════════════════════════════════════════════════════════════════════
def draw_metric_bar(frame, x, y, label, value, threshold, bar_w=120, is_high_bad=True):
    """
    Draw a small labelled bar for EAR / MAR / PERCLOS.
    is_high_bad=False → low value is bad (EAR), True → high value is bad (MAR, PERCLOS).
    """
    BAR_H = 10
    ratio = min(max(value / (threshold * 2), 0), 1)
    
    if is_high_bad:
        bad = value > threshold
    else:
        bad = value < threshold

    bar_color = (50, 50, 220) if bad else (50, 200, 100)
    fill_w = int(bar_w * ratio)

    # Background bar
    cv2.rectangle(frame, (x, y), (x + bar_w, y + BAR_H),
                  (60, 60, 60), -1)
    # Filled portion
    cv2.rectangle(frame, (x, y), (x + fill_w, y + BAR_H),
                  bar_color, -1)
    # Threshold tick
    tick_x = int(x + bar_w * 0.5)
    cv2.line(frame, (tick_x, y - 2), (tick_x, y + BAR_H + 2), (200, 200, 200), 1)
    # Label + value
    cv2.putText(frame, f"{label}: {value:.3f}",
                (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (220, 220, 220), 1, cv2.LINE_AA)


def draw_hud(frame, ear, mar, perclos, pitch, alerts):
    """Draw all metric bars and alert banners."""
    H, W = frame.shape[:2]
    panel_x = 10

    # ── metric bars (bottom-left panel) ──
    base_y = H - 120
    cv2.rectangle(frame, (0, base_y - 20), (200, H), (0, 0, 0), -1)
    cv2.addWeighted(frame[base_y-20:H, 0:200].copy(),
                    0.45, frame[base_y-20:H, 0:200], 0.55, 0,
                    frame[base_y-20:H, 0:200])

    draw_metric_bar(frame, panel_x, base_y,
                    "EAR", ear, EAR_THRESH, is_high_bad=False)
    draw_metric_bar(frame, panel_x, base_y + 30,
                    "MAR", mar, MAR_THRESH, is_high_bad=True)
    draw_metric_bar(frame, panel_x, base_y + 60,
                    "PERCLOS", perclos, PERCLOS_THRESH, is_high_bad=True)
    cv2.putText(frame, f"Pitch: {pitch:+.1f} deg",
                (panel_x, base_y + 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (50, 50, 220) if abs(pitch) > PITCH_THRESH else (220, 220, 220),
                1, cv2.LINE_AA)

    # ── alert banners (top) ──
    banner_y = 0
    for label, color in alerts:
        ov = frame.copy()
        cv2.rectangle(ov, (0, banner_y), (W, banner_y + 55), color, -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, label,
                    (15, banner_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (255, 255, 255), 2, cv2.LINE_AA)
        banner_y += 58


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
def run(camera=0, show_landmarks=True, show_fps=True):
    download_model()

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open camera {camera}")

    W_cap = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_cap = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Camera matrix (approximate, focal length = frame width)
    focal = W_cap
    cam_matrix   = np.array([[focal, 0, W_cap/2],
                              [0, focal, H_cap/2],
                              [0, 0, 1]], dtype=np.float64)
    dist_coeffs  = np.zeros((4,1), dtype=np.float64)

    # State
    ear_counter   = 0
    mar_counter   = 0
    pitch_counter = 0
    last_alert    = [0.0]   # mutable ref for alert()

    # PERCLOS: rolling deque of (timestamp, eye_closed_bool)
    perclos_buf   = deque()

    fps_t = time.time()
    fps_n = 0
    fps_v = 0.0

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("[INFO] Detector running. Press Q to quit.")
    print(f"       EAR thresh={EAR_THRESH}  MAR thresh={MAR_THRESH}  "
          f"PERCLOS thresh={PERCLOS_THRESH*100:.0f}%  Pitch thresh={PITCH_THRESH}°")

    with FaceLandmarker.create_from_options(options) as fm:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            H, W = frame.shape[:2]
            now  = time.time()

            # FPS
            fps_n += 1
            if now - fps_t >= 1.0:
                fps_v = fps_n / (now - fps_t)
                fps_n = 0
                fps_t = now

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res    = fm.detect(mp_img)

            ear_val   = 0.0
            mar_val   = 0.0
            pitch_val = 0.0
            alerts    = []

            if res.face_landmarks:
                lms = res.face_landmarks[0]

                # ── EAR ───────────────────────────────────────────────────
                le  = compute_ear(lms, LEFT_EYE,  W, H)
                re  = compute_ear(lms, RIGHT_EYE, W, H)
                ear_val = (le + re) / 2.0

                eye_closed = ear_val < EAR_THRESH
                if eye_closed:
                    ear_counter += 1
                    if ear_counter >= EAR_CONSEC:
                        alerts.append(("DROWSY  — eyes closing!", (0, 0, 180)))
                        alert(last_alert)
                else:
                    ear_counter = 0

                # ── PERCLOS ───────────────────────────────────────────────
                perclos_buf.append((now, eye_closed))
                cutoff = now - PERCLOS_WINDOW
                while perclos_buf and perclos_buf[0][0] < cutoff:
                    perclos_buf.popleft()
                if len(perclos_buf) > 5:
                    closed_count = sum(1 for _, c in perclos_buf if c)
                    perclos_val  = closed_count / len(perclos_buf)
                else:
                    perclos_val  = 0.0

                if perclos_val >= PERCLOS_THRESH and len(perclos_buf) > 30:
                    alerts.append(("FATIGUE — high PERCLOS!", (120, 0, 180)))
                    alert(last_alert)

                # ── MAR (yawn) ────────────────────────────────────────────
                mar_val = compute_mar(lms, W, H)
                if mar_val > MAR_THRESH:
                    mar_counter += 1
                    if mar_counter >= MAR_CONSEC:
                        alerts.append(("YAWN DETECTED!", (0, 120, 0)))
                        alert(last_alert)
                else:
                    mar_counter = 0

                # ── Head pitch ────────────────────────────────────────────
                pitch_val = compute_pitch(lms, W, H, cam_matrix, dist_coeffs)
                if abs(pitch_val) > PITCH_THRESH:
                    pitch_counter += 1
                    if pitch_counter >= PITCH_CONSEC:
                        alerts.append(("HEAD NODDING!", (180, 80, 0)))
                        alert(last_alert)
                else:
                    pitch_counter = 0

                # ── Eye / mouth landmark dots ─────────────────────────────
                if show_landmarks:
                    for i in LEFT_EYE + RIGHT_EYE:
                        cx, cy = int(lms[i].x * W), int(lms[i].y * H)
                        cv2.circle(frame, (cx, cy), 2, (255, 200, 0), -1)
                    for i in [78, 308, 82, 87, 13, 14, 312, 317]:
                        cx, cy = int(lms[i].x * W), int(lms[i].y * H)
                        cv2.circle(frame, (cx, cy), 2, (0, 200, 255), -1)

            else:
                perclos_val = 0.0

            # ── Draw HUD ──────────────────────────────────────────────────
            draw_hud(frame, ear_val, mar_val, perclos_val, pitch_val, alerts)

            if not alerts:
                cv2.putText(frame, "AWAKE", (15, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 2, cv2.LINE_AA)

            if show_fps:
                cv2.putText(frame, f"FPS {fps_v:.0f}",
                            (W - 90, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)

            cv2.imshow("Drowsiness Detector L1  [Q = quit]", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Stopped.")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Drowsiness Detector — Level 1")
    ap.add_argument("--camera",       type=int, default=0)
    ap.add_argument("--no-landmarks", action="store_true")
    ap.add_argument("--no-fps",       action="store_true")
    a  = ap.parse_args()
    run(a.camera, not a.no_landmarks, not a.no_fps)

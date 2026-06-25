"""
STEP 1 — Data Collection
========================
Records labelled eye-crop images from your webcam.

Usage:
    python collect_data.py --label awake   --samples 300
    python collect_data.py --label drowsy  --samples 300

Controls while recording:
    SPACE  → capture current frame
    A      → toggle auto-capture every N frames
    Q      → quit

Output layout:
    dataset/
      awake/   0001.png  0002.png ...
      drowsy/  0001.png  0002.png ...
"""

import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions
import numpy as np
import os
import argparse
import time

MODEL_PATH  = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")

# Eye indices for ROI crop
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
IMG_SIZE   = 64   # CNN input: 64×64 grayscale


def get_eye_roi(frame, lms, indices, W, H, pad=10):
    """Crop a padded bounding box around one eye, resize to IMG_SIZE."""
    pts = [(int(lms[i].x * W), int(lms[i].y * H)) for i in indices]
    xs  = [p[0] for p in pts]
    ys  = [p[1] for p in pts]
    x1  = max(0, min(xs) - pad)
    y1  = max(0, min(ys) - pad)
    x2  = min(W, max(xs) + pad)
    y2  = min(H, max(ys) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame[y1:y2, x1:x2]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    return roi


def collect(label, target_samples, camera=0):
    out_dir = os.path.join(DATASET_DIR, label)
    os.makedirs(out_dir, exist_ok=True)

    # Count existing samples
    existing = len([f for f in os.listdir(out_dir) if f.endswith(".png")])
    count    = existing
    print(f"[INFO] Collecting '{label}' — already have {existing}, target {target_samples}")
    print("[INFO] SPACE=capture  A=auto-capture  Q=quit")

    cap  = cv2.VideoCapture(camera)
    auto = False
    auto_interval = 5   # capture every N frames
    frame_n = 0

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with FaceLandmarker.create_from_options(options) as fm:
        while count < target_samples:
            ok, frame = cap.read()
            if not ok:
                continue

            H, W = frame.shape[:2]
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res  = fm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

            display = frame.copy()
            saved   = False

            if res.face_landmarks:
                lms    = res.face_landmarks[0]
                l_roi  = get_eye_roi(frame, lms, LEFT_EYE,  W, H)
                r_roi  = get_eye_roi(frame, lms, RIGHT_EYE, W, H)

                # Draw eye boxes
                for indices in [LEFT_EYE, RIGHT_EYE]:
                    pts = [(int(lms[i].x*W), int(lms[i].y*H)) for i in indices]
                    xs  = [p[0] for p in pts]
                    ys  = [p[1] for p in pts]
                    cv2.rectangle(display,
                                  (min(xs)-10, min(ys)-10),
                                  (max(xs)+10, max(ys)+10),
                                  (0, 255, 0), 1)

                should_save = (not auto and False) or (auto and frame_n % auto_interval == 0)

                if should_save and l_roi is not None and r_roi is not None:
                    for eye_roi, side in [(l_roi, "L"), (r_roi, "R")]:
                        fname = f"{count:04d}_{side}.png"
                        cv2.imwrite(os.path.join(out_dir, fname), eye_roi)
                    count += 1
                    saved = True

            # HUD
            prog = int((count / target_samples) * 200)
            cv2.rectangle(display, (10, H-30), (210, H-15), (50,50,50), -1)
            cv2.rectangle(display, (10, H-30), (10+prog, H-15), (0,200,100), -1)
            cv2.putText(display, f"{label.upper()}  {count}/{target_samples}",
                        (10, H-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(display, f"AUTO: {'ON' if auto else 'OFF'}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,200,100) if auto else (100,100,200), 1)
            if saved:
                cv2.putText(display, "SAVED", (W-90, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,200), 2)

            cv2.imshow(f"Collect — {label}", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" ") and res.face_landmarks:
                lms   = res.face_landmarks[0]
                l_roi = get_eye_roi(frame, lms, LEFT_EYE,  W, H)
                r_roi = get_eye_roi(frame, lms, RIGHT_EYE, W, H)
                if l_roi is not None and r_roi is not None:
                    for eye_roi, side in [(l_roi, "L"), (r_roi, "R")]:
                        fname = f"{count:04d}_{side}.png"
                        cv2.imwrite(os.path.join(out_dir, fname), eye_roi)
                    count += 1
            elif key == ord("a"):
                auto = not auto

            frame_n += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Saved {count} samples to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label",   required=True, choices=["awake", "drowsy"])
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--camera",  type=int, default=0)
    a = ap.parse_args()
    collect(a.label, a.samples, a.camera)

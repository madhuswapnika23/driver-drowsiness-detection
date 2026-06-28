"""
Driver Drowsiness Detection — Streamlit Cloud Deployment
Author: Madhu Swapnika G et al. | CSE303, SRM University-AP
Uses MediaPipe Tasks API (mediapipe >= 0.10.x)
"""

import streamlit as st
import mediapipe as mp
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions
import numpy as np
import av
import time
import os
import urllib.request
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ── landmark indices ───────────────────────────────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# ── model setup ────────────────────────────────────────────────────────────────
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = "/tmp/face_landmarker.task"

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading face landmark model (first run)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

ensure_model()

# ── thresholds ─────────────────────────────────────────────────────────────────
EAR_THRESH    = 0.25
EAR_CONSEC    = 35
MAR_THRESH    = 0.65
FUSION_THRESH = 0.65

# ── helpers ────────────────────────────────────────────────────────────────────
def dist(a, b):
    return np.linalg.norm(np.subtract(a, b))

def compute_ear(lms, idx, W, H):
    p = [(lms[i].x * W, lms[i].y * H) for i in idx]
    return (dist(p[1], p[5]) + dist(p[2], p[4])) / (2.0 * dist(p[0], p[3]))

def compute_mar(lms, W, H):
    def p(i): return np.array((lms[i].x * W, lms[i].y * H))
    v = (np.linalg.norm(p(82)-p(87)) + np.linalg.norm(p(13)-p(14)) +
         np.linalg.norm(p(312)-p(317)))
    return v / (2.0 * np.linalg.norm(p(78)-p(308)) + 1e-6)

def get_roi(frame, lms, idx, W, H, pad=10):
    import cv2
    pts = [(int(lms[i].x*W), int(lms[i].y*H)) for i in idx]
    xs  = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x1,y1 = max(0,min(xs)-pad), max(0,min(ys)-pad)
    x2,y2 = min(W,max(xs)+pad), min(H,max(ys)+pad)
    if x2<=x1 or y2<=y1: return None
    roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return cv2.resize(roi, (64,64)).astype(np.float32)/255.

# ── video processor ────────────────────────────────────────────────────────────
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.counter  = 0
        self.drowsy   = False
        self.ear_val  = 0.0
        self.mar_val  = 0.0
        self.score    = 0.0

        opts = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._fm = FaceLandmarker.create_from_options(opts)

    def recv(self, frame):
        import cv2
        img    = frame.to_ndarray(format="bgr24")
        H, W   = img.shape[:2]
        rgb    = img[:, :, ::-1]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res    = self._fm.detect(mp_img)

        self.drowsy = False

        if res.face_landmarks:
            lms = res.face_landmarks[0]

            le = compute_ear(lms, LEFT_EYE,  W, H)
            re = compute_ear(lms, RIGHT_EYE, W, H)
            self.ear_val = round((le + re) / 2.0, 3)
            self.mar_val = round(compute_mar(lms, W, H), 3)

            ear_s = max(0., 1. - self.ear_val/EAR_THRESH) if self.ear_val < EAR_THRESH*1.5 else 0.
            mar_s = min(1., self.mar_val/MAR_THRESH)
            self.score = round(0.7*ear_s + 0.3*mar_s, 3)

            if self.score >= FUSION_THRESH:
                self.counter += 1
                if self.counter >= EAR_CONSEC:
                    self.drowsy = True
            else:
                self.counter = 0

            for i in LEFT_EYE + RIGHT_EYE:
                cv2.circle(img, (int(lms[i].x*W), int(lms[i].y*H)), 2, (255,200,0), -1)

            if self.drowsy:
                cv2.rectangle(img, (0,0), (W,65), (0,0,180), -1)
                cv2.putText(img, f"DROWSY! score={self.score}", (15,48),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2)
            else:
                cv2.putText(img, "AWAKE", (15,48),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,200,0), 2)

            cv2.putText(img, f"EAR:{self.ear_val}  MAR:{self.mar_val}",
                        (10,H-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        else:
            import cv2
            cv2.putText(img, "No face detected", (15,48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80,80,0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Driver Drowsiness Detector", page_icon="👁️", layout="wide")
st.title("👁️ Driver Drowsiness Detection System")
st.caption("Real-time detection using MediaPipe FaceMesh + EAR | CSE303 ML Project — SRM University-AP")

with st.sidebar:
    st.markdown("### Team")
    st.markdown("""
    - Madhu Swapnika G  
    - P. Nikitha  
    - Rakshitha Joycey  
    - Hema Latha U  
    - G. Sowjanya  

    **Course:** CSE303 Machine Learning  
    **University:** SRM University-AP
    """)
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
    1. MediaPipe detects 468 face landmarks  
    2. EAR computed from 6 eye points  
    3. MAR computed from mouth points  
    4. Fusion score triggers alert  
    """)

col_cam, col_stats = st.columns([3, 1])

with col_stats:
    st.markdown("### Live Readings")
    status_ph = st.empty()
    ear_ph    = st.empty()
    mar_ph    = st.empty()
    score_ph  = st.empty()
    st.markdown("---")
    st.markdown("### Alert Log")
    log_ph    = st.empty()
    alert_log = []

with col_cam:
    st.info("Allow camera access when your browser asks, then click START.")
    ctx = webrtc_streamer(
        key="drowsiness",
        video_processor_factory=DrowsinessProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
            ]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

if ctx.video_processor:
    proc = ctx.video_processor
    while True:
        if proc.drowsy:
            status_ph.error("🚨 DROWSY ALERT!")
            ts = time.strftime("%H:%M:%S")
            alert_log.append(f"{ts} — DROWSY (score {proc.score})")
            log_ph.dataframe({"Alerts": alert_log[-10:]})
        else:
            status_ph.success("✅ AWAKE")

        ear_ph.metric("EAR", proc.ear_val, delta=round(proc.ear_val - EAR_THRESH, 3))
        mar_ph.metric("MAR", proc.mar_val)
        score_ph.progress(min(proc.score, 1.0), text=f"Fusion Score: {proc.score}")
        time.sleep(0.4)
        st.rerun()
"""
Driver Drowsiness Detection — Streamlit Cloud
Uses OpenCV DNN face detection + EAR (no mediapipe on cloud)
Author: Madhu Swapnika G et al. | CSE303, SRM University-AP
"""

import streamlit as st
import numpy as np
import av
import time
import os
import urllib.request
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ── eye landmark indices for dlib-style 68-point (via OpenCV) ─────────────────
# We use OpenCV's face detector + eye cascade for cloud compatibility
EAR_THRESH    = 0.25
EAR_CONSEC    = 30
MAR_THRESH    = 0.65
FUSION_THRESH = 0.60

# ── download haar cascades at runtime ─────────────────────────────────────────
CASCADE_BASE = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
EYE_CASCADE_PATH  = "/tmp/haarcascade_eye.xml"
FACE_CASCADE_PATH = "/tmp/haarcascade_frontalface_default.xml"

def ensure_cascades():
    for url, path in [
        (CASCADE_BASE + "haarcascade_eye.xml",               EYE_CASCADE_PATH),
        (CASCADE_BASE + "haarcascade_frontalface_default.xml", FACE_CASCADE_PATH),
    ]:
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path)

ensure_cascades()

# ── helpers ────────────────────────────────────────────────────────────────────
def dist(a, b):
    return np.linalg.norm(np.subtract(a, b))

def eye_openness_ratio(eye_region_h, eye_region_w):
    """Approximate EAR from eye bounding box aspect ratio."""
    if eye_region_w == 0:
        return 0.3
    return eye_region_h / eye_region_w

# ── video processor ────────────────────────────────────────────────────────────
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        import cv2
        self.counter      = 0
        self.drowsy       = False
        self.ear_val      = 0.3
        self.score        = 0.0
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        self.eye_cascade  = cv2.CascadeClassifier(EYE_CASCADE_PATH)

    def recv(self, frame):
        import cv2
        img  = frame.to_ndarray(format="bgr24")
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)   # improve low-light

        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))

        self.drowsy = False

        if len(faces) > 0:
            # Take largest face
            fx, fy, fw, fh = max(faces, key=lambda f: f[2]*f[3])
            face_gray = gray[fy:fy+fh, fx:fx+fw]
            face_img  = img[fy:fy+fh, fx:fx+fw]

            cv2.rectangle(img, (fx,fy), (fx+fw,fy+fh), (0,200,0), 1)

            eyes = self.eye_cascade.detectMultiScale(
                face_gray, scaleFactor=1.05, minNeighbors=3, minSize=(20,20))

            ear_vals = []
            for (ex, ey, ew, eh) in eyes[:2]:
                ratio = round(eh / max(ew, 1), 3)
                ear_vals.append(ratio)
                color = (50,50,220) if ratio < EAR_THRESH else (255,200,0)
                cv2.rectangle(face_img, (ex,ey), (ex+ew,ey+eh), color, 1)

            if ear_vals:
                self.ear_val = round(float(np.mean(ear_vals)), 3)
                ear_s = max(0., 1. - self.ear_val/EAR_THRESH) if self.ear_val < EAR_THRESH*1.5 else 0.
                self.score = round(ear_s, 3)

                if self.score >= FUSION_THRESH:
                    self.counter += 1
                    if self.counter >= EAR_CONSEC:
                        self.drowsy = True
                else:
                    self.counter = 0
            else:
                # No eyes detected — possible drowsiness
                self.counter += 1
                self.ear_val = 0.0
                self.score   = 0.8
                if self.counter >= EAR_CONSEC:
                    self.drowsy = True

            # Status overlay
            if self.drowsy:
                cv2.rectangle(img, (0,0), (W,65), (0,0,180), -1)
                cv2.putText(img, f"DROWSY! score={self.score:.2f}",
                            (15,48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2)
            else:
                cv2.putText(img, "AWAKE",
                            (15,48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,200,0), 2)

            cv2.putText(img, f"EAR:{self.ear_val}  Score:{self.score}",
                        (10,H-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        else:
            self.counter = 0
            cv2.putText(img, "No face detected",
                        (15,48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80,80,0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Driver Drowsiness Detector",
    page_icon="👁️", layout="wide")

st.title("👁️ Driver Drowsiness Detection System")
st.caption("Real-time detection using OpenCV + EAR | CSE303 ML Project — SRM University-AP")

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
1. OpenCV detects face using Haar cascade  
2. Eye regions located within face  
3. Eye aspect ratio (EAR) computed  
4. EAR < 0.25 for 30 frames → DROWSY alert  
    """)
    st.markdown("---")
    st.markdown("### Thresholds")
    st.code(f"EAR threshold  : {EAR_THRESH}\nFrames to alert: {EAR_CONSEC}\nFusion score   : {FUSION_THRESH}")

col_cam, col_stats = st.columns([3, 1])

with col_stats:
    st.markdown("### Live Readings")
    status_ph = st.empty()
    ear_ph    = st.empty()
    score_ph  = st.empty()
    st.markdown("---")
    st.markdown("### Alert Log")
    log_ph    = st.empty()
    alert_log = []

with col_cam:
    st.info("Click START and allow camera access when your browser asks.")
    ctx = webrtc_streamer(
        key="drowsiness",
        video_processor_factory=DrowsinessProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]}),
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

        ear_ph.metric("EAR", proc.ear_val,
                      delta=round(proc.ear_val - EAR_THRESH, 3))
        score_ph.progress(min(proc.score, 1.0),
                          text=f"Fusion Score: {proc.score}")
        time.sleep(0.4)
        st.rerun()
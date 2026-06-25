"""
Driver Drowsiness Detection — Streamlit web app
Deploy: streamlit run app.py
Cloud:  share.streamlit.io  (free)
"""

import streamlit as st
import mediapipe as mp
import numpy as np
import av
import time
import threading

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ── constants ──────────────────────────────────────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

mp_face = mp.solutions.face_mesh


def _dist(a, b):
    return np.linalg.norm(np.subtract(a, b))


def compute_ear(lms, indices, W, H):
    p = [(lms[i].x * W, lms[i].y * H) for i in indices]
    return (_dist(p[1], p[5]) + _dist(p[2], p[4])) / (2.0 * _dist(p[0], p[3]))


# ── video processor ────────────────────────────────────────────────────────────
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.counter  = 0
        self.drowsy   = False
        self.ear_val  = 0.0
        self.thresh   = 0.25
        self.frames   = 20
        self._fm = mp_face.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame):
        import cv2
        img  = frame.to_ndarray(format="bgr24")
        H, W = img.shape[:2]
        res  = self._fm.process(img[:, :, ::-1])

        self.drowsy = False

        if res.multi_face_landmarks:
            lms = res.multi_face_landmarks[0].landmark
            le  = compute_ear(lms, LEFT_EYE,  W, H)
            re  = compute_ear(lms, RIGHT_EYE, W, H)
            self.ear_val = round((le + re) / 2, 3)

            for i in LEFT_EYE + RIGHT_EYE:
                cx, cy = int(lms[i].x * W), int(lms[i].y * H)
                cv2.circle(img, (cx, cy), 2, (255, 200, 0), -1)

            if self.ear_val < self.thresh:
                self.counter += 1
                if self.counter >= self.frames:
                    self.drowsy = True
                    cv2.rectangle(img, (0,0), (W, 65), (0,0,180), -1)
                    cv2.putText(img, "DROWSY!", (15, 48),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,0,255), 2)
            else:
                self.counter = 0

            cv2.putText(img, f"EAR: {self.ear_val}", (10, H-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        else:
            cv2.putText(img, "No face detected", (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100,100,0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ── UI ─────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Drowsiness Detector", page_icon="👁️", layout="wide")

st.title("👁️ Driver Drowsiness Detection")
st.caption("MediaPipe FaceMesh + Eye Aspect Ratio (EAR) — CSE303 ML Project, SRM-AP")

col_cam, col_info = st.columns([3, 1])

with col_info:
    st.markdown("### Live Status")
    status_ph = st.empty()
    ear_ph    = st.empty()

    st.divider()
    st.markdown("### Tune Parameters")
    thresh_val  = st.slider("EAR threshold",  0.15, 0.35, 0.25, 0.01,
                            help="Lower → harder to trigger. Default: 0.25")
    frames_val  = st.slider("Frames to alert", 5,   40,   20,   1,
                            help="How many consecutive closed frames = drowsy")

    st.divider()
    st.markdown("""
**How it works**

1. FaceMesh finds 468 face landmarks  
2. 6 points per eye → EAR formula  
3. EAR < threshold for N frames → 🚨  
4. Alert resets when eyes open  

**EAR formula**  
`(‖P₂-P₆‖ + ‖P₃-P₅‖) / (2‖P₁-P₄‖)`
    """)

with col_cam:
    ctx = webrtc_streamer(
        key="drowsy",
        video_processor_factory=DrowsinessProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# ── live status refresh ────────────────────────────────────────────────────────
if ctx.video_processor:
    proc = ctx.video_processor
    proc.thresh = thresh_val
    proc.frames = frames_val
    while True:
        if proc.drowsy:
            status_ph.error("⚠️ DROWSY — Wake up!")
        elif proc.ear_val > 0:
            status_ph.success("✅ Awake")
        else:
            status_ph.info("👁️ Waiting for face…")
        ear_ph.metric("EAR", proc.ear_val,
                      delta=round(proc.ear_val - thresh_val, 3),
                      delta_color="normal")
        time.sleep(0.4)
        st.rerun()

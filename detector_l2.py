"""
STEP 3 — Level 2 Detector
==========================
Runs the trained CNN model alongside the Level 1 signals (EAR, MAR,
PERCLOS, head pose) and fuses them into one confidence score.

Falls back to EAR-only mode if no trained model is found.

Usage:
    python detector_l2.py
    python detector_l2.py --mode cnn       # CNN only (fast)
    python detector_l2.py --mode fusion    # CNN + EAR + MAR + PERCLOS (default)
    python detector_l2.py --mode tflite    # TFLite model (edge-optimised)
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions
import time, threading, os, sys, argparse, json
from collections import deque

# ── optional sound ─────────────────────────────────────────────────────────────
try:
    from playsound import playsound; _SOUND = "playsound"
except ImportError:
    try:
        import winsound; _SOUND = "winsound"
    except ImportError:
        _SOUND = "beep"

BASE_DIR    = os.path.dirname(__file__)
MODEL_DIR   = os.path.join(BASE_DIR, "model")
MP_MODEL    = os.path.join(BASE_DIR, "face_landmarker.task")
SOUND_PATH  = os.path.join(BASE_DIR, "sounds", "alert.wav")
META_PATH   = os.path.join(MODEL_DIR, "meta.json")
CNN_PATH    = os.path.join(MODEL_DIR, "cnn_eye.keras")
TFLITE_PATH = os.path.join(MODEL_DIR, "drowsiness.tflite")

# ── landmark indices ────────────────────────────────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

MODEL_POINTS_3D = np.array([
    (0.0,0.0,0.0),(0.0,-330.0,-65.0),(-225.0,170.0,-135.0),
    (225.0,170.0,-135.0),(-150.0,-150.0,-125.0),(150.0,-150.0,-125.0)
], dtype=np.float64)
POSE_IDS = [1,152,33,263,61,291]

# ── thresholds ─────────────────────────────────────────────────────────────────
EAR_THRESH      = 0.25
EAR_CONSEC      = 35
MAR_THRESH      = 0.65
MAR_CONSEC      = 15
PERCLOS_WINDOW  = 60
PERCLOS_THRESH  = 0.30
PITCH_THRESH    = 20.0
PITCH_CONSEC    = 30
CNN_THRESH      = 0.55   # CNN drowsy probability threshold
FUSION_THRESH   = 0.65   # weighted fusion score threshold
COOLDOWN_SEC    = 3
IMG_SIZE        = 64


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS  (same as Level 1)
# ══════════════════════════════════════════════════════════════════════════════
def dist(a,b): return np.linalg.norm(np.subtract(a,b))

def to_px(lm,i,W,H): return (lm[i].x*W, lm[i].y*H)

def compute_ear(lms,idx,W,H):
    p=[to_px(lms,i,W,H) for i in idx]
    return (dist(p[1],p[5])+dist(p[2],p[4]))/(2.*dist(p[0],p[3]))

def compute_mar(lms,W,H):
    def p(i): return np.array(to_px(lms,i,W,H))
    v=(np.linalg.norm(p(82)-p(87))+np.linalg.norm(p(13)-p(14))+
       np.linalg.norm(p(312)-p(317)))
    h=np.linalg.norm(p(78)-p(308))
    return v/(2.*h+1e-6)

def compute_pitch(lms,W,H,cm,dc):
    pts=np.array([to_px(lms,i,W,H) for i in POSE_IDS],dtype=np.float64)
    ok,rv,_=cv2.solvePnP(MODEL_POINTS_3D,pts,cm,dc,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return 0.
    rm,_=cv2.Rodrigues(rv)
    sy=np.sqrt(rm[0,0]**2+rm[1,0]**2)
    return float(np.degrees(np.arctan2(-rm[2,0],sy)))

def get_eye_roi(frame,lms,idx,W,H,pad=10):
    pts=[(int(lms[i].x*W),int(lms[i].y*H)) for i in idx]
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    x1,y1=max(0,min(xs)-pad),max(0,min(ys)-pad)
    x2,y2=min(W,max(xs)+pad),min(H,max(ys)+pad)
    if x2<=x1 or y2<=y1: return None
    roi=cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY)
    return cv2.resize(roi,(IMG_SIZE,IMG_SIZE)).astype(np.float32)/255.


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════
class ModelBackend:
    def __init__(self, mode):
        self.mode   = mode
        self.model  = None
        self.interp = None   # TFLite interpreter
        self._load(mode)

    def _load(self, mode):
        if mode in ("cnn","fusion") and os.path.exists(CNN_PATH):
            import tensorflow as tf
            self.model = tf.keras.models.load_model(CNN_PATH)
            print(f"[MODEL] Loaded CNN from {CNN_PATH}")
        elif mode == "tflite" and os.path.exists(TFLITE_PATH):
            import tensorflow as tf
            self.interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
            self.interp.allocate_tensors()
            self.in_idx  = self.interp.get_input_details()[0]["index"]
            self.out_idx = self.interp.get_output_details()[0]["index"]
            print(f"[MODEL] Loaded TFLite from {TFLITE_PATH}")
        else:
            print("[MODEL] No trained model found — using EAR-only fallback.")
            print("        Run collect_data.py then train_model.py to enable CNN.")
            self.mode = "ear_only"

    def predict(self, roi):
        """Returns drowsy probability 0-1 for one eye ROI (64,64) float32."""
        if self.mode == "ear_only" or roi is None:
            return None
        inp = roi[np.newaxis, ..., np.newaxis]   # (1,64,64,1)
        if self.mode == "tflite":
            self.interp.set_tensor(self.in_idx, inp)
            self.interp.invoke()
            return float(self.interp.get_tensor(self.out_idx)[0][0])
        else:
            return float(self.model.predict(inp, verbose=0)[0][0])


# ══════════════════════════════════════════════════════════════════════════════
# FUSION SCORER
# ══════════════════════════════════════════════════════════════════════════════
def fusion_score(ear, mar, perclos, pitch, cnn_prob):
    """
    Weighted combination of all signals → single drowsiness score 0-1.
    Weights tuned so EAR + CNN together dominate.
    """
    ear_score     = max(0., 1. - ear/EAR_THRESH) if ear < EAR_THRESH*1.5 else 0.
    mar_score     = min(1., max(0., (mar - MAR_THRESH)/(1. - MAR_THRESH)))
    perclos_score = min(1., perclos / PERCLOS_THRESH)
    pitch_score   = min(1., abs(pitch) / (PITCH_THRESH*2))
    cnn_score     = cnn_prob if cnn_prob is not None else ear_score

    w = {"ear":0.25, "cnn":0.35, "perclos":0.20, "mar":0.10, "pitch":0.10}
    score = (w["ear"]*ear_score + w["cnn"]*cnn_score +
             w["perclos"]*perclos_score + w["mar"]*mar_score +
             w["pitch"]*pitch_score)
    return round(float(score), 3)


# ══════════════════════════════════════════════════════════════════════════════
# HUD
# ══════════════════════════════════════════════════════════════════════════════
def draw_score_bar(frame, score, x=10, y=None):
    H,W = frame.shape[:2]
    if y is None: y = H-20
    bar_w = 200
    filled = int(bar_w * score)
    color  = (50,200,50) if score<0.4 else (50,150,200) if score<0.6 else (50,50,220)
    cv2.rectangle(frame,(x,y),(x+bar_w,y+12),(40,40,40),-1)
    cv2.rectangle(frame,(x,y),(x+filled,y+12),color,-1)
    cv2.putText(frame,f"Fusion: {score:.2f}",(x,y-5),
                cv2.FONT_HERSHEY_SIMPLEX,0.45,(220,220,220),1)

def draw_signals(frame, ear, mar, perclos, pitch, cnn_p, mode):
    H,W = frame.shape[:2]
    panel_h = 120
    ov = frame.copy()
    cv2.rectangle(ov,(0,H-panel_h),(215,H),(0,0,0),-1)
    cv2.addWeighted(ov,0.45,frame,0.55,0,frame)
    base = H-panel_h+15
    lines = [
        f"EAR:     {ear:.3f}  {'<< LOW' if ear<EAR_THRESH else ''}",
        f"MAR:     {mar:.3f}  {'<< YAWN' if mar>MAR_THRESH else ''}",
        f"PERCLOS: {perclos*100:.1f}%",
        f"Pitch:   {pitch:+.1f}°",
        f"CNN:     {f'{cnn_p:.2f}' if cnn_p is not None else 'N/A'}  [{mode}]",
    ]
    for i,txt in enumerate(lines):
        col = (100,100,255) if (
            ("LOW" in txt or "YAWN" in txt) or
            (i==2 and perclos>=PERCLOS_THRESH) or
            (i==3 and abs(pitch)>=PITCH_THRESH) or
            (i==4 and cnn_p is not None and cnn_p>=CNN_THRESH)
        ) else (200,200,200)
        cv2.putText(frame,txt,(8,base+i*22),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,col,1,cv2.LINE_AA)

def _beep():
    if _SOUND=="playsound" and os.path.exists(SOUND_PATH):
        try: playsound(SOUND_PATH,block=False); return
        except: pass
    if _SOUND=="winsound":
        try: winsound.Beep(1000,500); return
        except: pass
    print("\a",end="",flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
def run(camera=0, mode="fusion"):
    backend = ModelBackend(mode)

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened(): sys.exit(f"[ERROR] Cannot open camera {camera}")

    W_cap = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_cap = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    focal  = W_cap
    cm     = np.array([[focal,0,W_cap/2],[0,focal,H_cap/2],[0,0,1]],dtype=np.float64)
    dc     = np.zeros((4,1),dtype=np.float64)

    ear_ctr=mar_ctr=pitch_ctr=0
    last_alert=[0.]
    perclos_buf=deque()
    fps_t=time.time(); fps_n=0; fps_v=0.

    mp_opts = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MP_MODEL),
        running_mode=RunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print(f"[INFO] Level 2 Detector running | mode={backend.mode} | Q=quit")

    with FaceLandmarker.create_from_options(mp_opts) as fm:
        while True:
            ok,frame = cap.read()
            if not ok: continue
            H,W = frame.shape[:2]
            now = time.time()

            fps_n+=1
            if now-fps_t>=1.:
                fps_v=fps_n/(now-fps_t); fps_n=0; fps_t=now

            rgb    = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
            res    = fm.detect(mp_img)

            ear_v=0.; mar_v=0.; pitch_v=0.; cnn_p=None; score=0.
            alerts=[]

            if res.face_landmarks:
                lms = res.face_landmarks[0]

                # geometry signals
                le   = compute_ear(lms,LEFT_EYE,W,H)
                re   = compute_ear(lms,RIGHT_EYE,W,H)
                ear_v= (le+re)/2.
                mar_v= compute_mar(lms,W,H)
                pitch_v=compute_pitch(lms,W,H,cm,dc)

                # PERCLOS
                eye_closed = ear_v < EAR_THRESH
                perclos_buf.append((now,eye_closed))
                cutoff = now-PERCLOS_WINDOW
                while perclos_buf and perclos_buf[0][0]<cutoff: perclos_buf.popleft()
                perclos_v=(sum(1 for _,c in perclos_buf if c)/len(perclos_buf)
                           if len(perclos_buf)>5 else 0.)

                # CNN prediction (average both eyes)
                l_roi = get_eye_roi(frame,lms,LEFT_EYE,W,H)
                r_roi = get_eye_roi(frame,lms,RIGHT_EYE,W,H)
                probs=[p for p in [backend.predict(l_roi),backend.predict(r_roi)]
                       if p is not None]
                cnn_p = float(np.mean(probs)) if probs else None

                # Fusion score
                score = fusion_score(ear_v,mar_v,perclos_v,pitch_v,cnn_p)

                # Alert logic
                if backend.mode=="ear_only":
                    if ear_ctr>=EAR_CONSEC:   alerts.append(("DROWSY — eyes closing!",(0,0,180)))
                    if mar_ctr>=MAR_CONSEC:    alerts.append(("YAWN DETECTED!",(0,120,0)))
                    if pitch_ctr>=PITCH_CONSEC:alerts.append(("HEAD NODDING!",(180,80,0)))
                    if eye_closed: ear_ctr+=1
                    else:          ear_ctr=0
                    if mar_v>MAR_THRESH: mar_ctr+=1
                    else:                mar_ctr=0
                    if abs(pitch_v)>PITCH_THRESH: pitch_ctr+=1
                    else:                         pitch_ctr=0
                else:
                    if score>=FUSION_THRESH:
                        alerts.append((f"DROWSY! (score {score:.2f})",(0,0,180)))

                if alerts and now-last_alert[0]>COOLDOWN_SEC:
                    last_alert[0]=now
                    threading.Thread(target=_beep,daemon=True).start()

                # landmarks
                for i in LEFT_EYE+RIGHT_EYE:
                    cv2.circle(frame,(int(lms[i].x*W),int(lms[i].y*H)),2,(255,200,0),-1)

            else:
                perclos_v=0.

            # draw HUD
            draw_signals(frame,ear_v,mar_v,
                         perclos_v if res.face_landmarks else 0.,
                         pitch_v,cnn_p,backend.mode)
            draw_score_bar(frame,score)

            banner_y=0
            for lbl,col in alerts:
                ov=frame.copy()
                cv2.rectangle(ov,(0,banner_y),(W,banner_y+55),col,-1)
                cv2.addWeighted(ov,0.55,frame,0.45,0,frame)
                cv2.putText(frame,lbl,(15,banner_y+40),
                            cv2.FONT_HERSHEY_SIMPLEX,1.1,(255,255,255),2,cv2.LINE_AA)
                banner_y+=58

            if not alerts:
                cv2.putText(frame,"AWAKE",(15,40),
                            cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,200,0),2,cv2.LINE_AA)

            cv2.putText(frame,f"FPS {fps_v:.0f}",(W-90,28),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(160,160,160),1)

            cv2.imshow("Drowsiness L2  [Q=quit]",frame)
            if cv2.waitKey(1)&0xFF==ord("q"): break

    cap.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--camera",type=int,default=0)
    ap.add_argument("--mode",choices=["cnn","fusion","tflite","ear_only"],
                    default="fusion")
    a=ap.parse_args()
    run(a.camera,a.mode)
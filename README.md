# Driver Drowsiness Detection System

**CSE303: Machine Learning | SRM University-AP**  
Madhu Swapnika G · P. Nikitha · Rakshitha Joycey · Hema Latha · G. Sowjanya

Real-time drowsiness detection using **MediaPipe FaceMesh** + **Eye Aspect Ratio (EAR)** — no deep learning required, runs on a laptop webcam at 20–30 FPS.

---

## Quick Start (Local / Desktop)

### 1. Prerequisites

```bash
# Python 3.10 is required (especially on Windows for mediapipe)
python --version   # should show 3.10.x
```

### 2. Clone / download project

```bash
git clone https://github.com/Madhuswapnika23/drowsiness-detector
cd drowsiness-detector
```

### 3. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

> **Windows note:** If `playsound` fails, install with:
> `pip install playsound==1.3.0 PyObjC` (Mac) or just skip it — the system will fall back to a beep.

### 5. Generate alert sound (first time only)

```bash
python sounds/generate_alert.py
```

### 6. Run the detector

```bash
python detector.py
```

**Options:**
```bash
python detector.py --camera 1          # use external webcam
python detector.py --no-landmarks      # hide eye dots
python detector.py --no-fps            # hide FPS counter
```

Press **Q** to quit.

---

## Deploy as a Web App (Streamlit)

### Install extra dependencies

```bash
pip install streamlit streamlit-webrtc
```

### Run locally

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

### Deploy to Streamlit Cloud (free, public URL)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select repo → `app.py` → click **Deploy**
5. Done — your app gets a public URL like `https://yourname-drowsiness-detector.streamlit.app`

> **Note:** Streamlit Cloud grants camera access over HTTPS automatically.

---

## How It Works

```
Webcam frame
    └── BGR → RGB conversion
    └── MediaPipe FaceMesh → 468 landmarks
    └── Extract 6 eye points per eye (LEFT_EYE, RIGHT_EYE indices)
    └── Compute EAR = (||P2-P6|| + ||P3-P5||) / (2 × ||P1-P4||)
    └── EAR < 0.25 for 20 consecutive frames?
            YES → Display "DROWSY!" + play alert.wav
            NO  → Display "AWAKE"
```

**Key constants** (editable in `detector.py`):

| Constant | Default | Meaning |
|---|---|---|
| `EAR_THRESHOLD` | `0.25` | Below this → eye considered closed |
| `CONSEC_FRAMES` | `20` | Frames of closed eye before alert |
| `alert_cooldown` | `3 s` | Gap between repeated alerts |

---

## Project Structure

```
drowsiness-detector/
├── detector.py          ← main desktop script
├── app.py               ← Streamlit web app
├── requirements.txt
├── sounds/
│   ├── generate_alert.py  ← run once to create alert.wav
│   └── alert.wav          ← generated alert sound
└── README.md
```

---

## Extending the Project (Future Scope)

| Feature | How |
|---|---|
| Yawn detection | Compute Mouth Aspect Ratio (MAR) using mouth landmark indices |
| Head pose estimation | Use MediaPipe `refine_landmarks` + solvePnP |
| Night vision / IR | Replace `VideoCapture(0)` with IR camera feed |
| CNN fallback | Train a small model on EAR sequences for harder cases |
| Mobile app | Port to Kivy or Flutter + TFLite |

---

## References

- Soukupová & Čech (2016) — Real-time Eye Blink Detection using Facial Landmarks  
- MediaPipe FaceMesh: https://google.github.io/mediapipe/  
- OpenCV: https://docs.opencv.org/

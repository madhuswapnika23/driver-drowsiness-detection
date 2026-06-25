"""
Generates alert.wav in this folder using only Python stdlib.
Run once: python sounds/generate_alert.py
"""
import wave, struct, math, os

OUTPUT = os.path.join(os.path.dirname(__file__), "alert.wav")

SAMPLE_RATE = 44100
DURATION    = 0.6       # seconds
FREQUENCY   = 1000      # Hz  (sharp beep)
AMPLITUDE   = 28000

samples = []
total   = int(SAMPLE_RATE * DURATION)
for i in range(total):
    # Fade in/out to avoid clicks
    t      = i / SAMPLE_RATE
    fade   = min(i, total - i, SAMPLE_RATE // 40) / (SAMPLE_RATE // 40)
    value  = int(AMPLITUDE * fade * math.sin(2 * math.pi * FREQUENCY * t))
    samples.append(struct.pack("<h", value))

with wave.open(OUTPUT, "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b"".join(samples))

print(f"[OK] Alert sound written to {OUTPUT}")

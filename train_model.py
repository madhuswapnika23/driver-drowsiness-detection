"""
STEP 2 — Train CNN + LSTM Model
================================
Trains a MobileNetV2-based CNN on your collected eye images,
then wraps it in an LSTM to learn temporal blink sequences,
and exports to both .keras and .tflite formats.

Usage:
    python train_model.py

Outputs:
    model/cnn_eye.keras       — CNN only (fast, per-frame)
    model/cnn_lstm.keras      — CNN + LSTM (temporal, more accurate)
    model/drowsiness.tflite   — TFLite export for mobile / edge
    model/training_plot.png   — accuracy / loss curves
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import json

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR   = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── config ─────────────────────────────────────────────────────────────────────
IMG_SIZE    = 64       # must match collect_data.py
SEQ_LEN     = 20       # LSTM looks at 20 consecutive frames
BATCH_SIZE  = 32
EPOCHS_CNN  = 20
EPOCHS_LSTM = 15
CLASSES     = ["awake", "drowsy"]   # index 0 = awake, 1 = drowsy


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATASET
# ══════════════════════════════════════════════════════════════════════════════
def load_dataset():
    X, y = [], []
    for label_idx, label in enumerate(CLASSES):
        folder = os.path.join(DATASET_DIR, label)
        if not os.path.exists(folder):
            print(f"[WARN] Missing folder: {folder}")
            continue
        files = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
        print(f"[DATA] {label}: {len(files)} images")
        for fname in files:
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32) / 255.0
            X.append(img)
            y.append(label_idx)

    X = np.array(X)[..., np.newaxis]   # (N, 64, 64, 1)
    y = np.array(y)
    print(f"[DATA] Total: {len(X)} images | awake={np.sum(y==0)} drowsy={np.sum(y==1)}")
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# 2. CNN MODEL  (MobileNetV2 backbone, grayscale→RGB trick)
# ══════════════════════════════════════════════════════════════════════════════
def build_cnn():
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    # Grayscale → 3-channel so MobileNetV2 works
    x = layers.Conv2D(3, 1, padding="same", use_bias=False)(inp)

    # MobileNetV2 backbone (pretrained on ImageNet)
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        alpha=0.35          # lightest variant — fast on CPU
    )
    base.trainable = False  # freeze backbone initially

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)   # 0=awake 1=drowsy

    model = models.Model(inp, out, name="cnn_eye")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model, base


# ══════════════════════════════════════════════════════════════════════════════
# 3. CNN + LSTM MODEL  (temporal sequence of eye states)
# ══════════════════════════════════════════════════════════════════════════════
def build_cnn_lstm(cnn_feature_extractor):
    """
    Wraps the CNN feature extractor in a TimeDistributed layer,
    then passes the sequence through a BiLSTM for temporal context.
    Input shape: (batch, SEQ_LEN, IMG_SIZE, IMG_SIZE, 1)
    """
    inp = layers.Input(shape=(SEQ_LEN, IMG_SIZE, IMG_SIZE, 1))
    x   = layers.TimeDistributed(cnn_feature_extractor)(inp)
    x   = layers.Bidirectional(layers.LSTM(32, return_sequences=False))(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inp, out, name="cnn_lstm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model


def make_feature_model(full_cnn):
    """Extract CNN up to the Dense(64) layer for use in LSTM."""
    return models.Model(
        inputs  = full_cnn.input,
        outputs = full_cnn.layers[-3].output,   # Dense(64) output
        name    = "cnn_features"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAKE SEQUENCES  (slide a window over sorted frames per class)
# ══════════════════════════════════════════════════════════════════════════════
def make_sequences(X, y, seq_len=SEQ_LEN):
    Xs, ys = [], []
    for label in [0, 1]:
        idxs = np.where(y == label)[0]
        imgs = X[idxs]
        for i in range(0, len(imgs) - seq_len, seq_len // 2):
            Xs.append(imgs[i:i+seq_len])
            ys.append(label)
    return np.array(Xs), np.array(ys)


# ══════════════════════════════════════════════════════════════════════════════
# 5. PLOT TRAINING CURVES
# ══════════════════════════════════════════════════════════════════════════════
def plot_history(histories, names):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["steelblue", "tomato"]
    for hist, name, c in zip(histories, names, colors):
        axes[0].plot(hist.history["accuracy"],     label=f"{name} train", color=c, linestyle="-")
        axes[0].plot(hist.history["val_accuracy"], label=f"{name} val",   color=c, linestyle="--")
        axes[1].plot(hist.history["loss"],         label=f"{name} train", color=c, linestyle="-")
        axes[1].plot(hist.history["val_loss"],     label=f"{name} val",   color=c, linestyle="--")
    axes[0].set_title("Accuracy");  axes[0].legend(); axes[0].set_xlabel("Epoch")
    axes[1].set_title("Loss");      axes[1].legend(); axes[1].set_xlabel("Epoch")
    plt.tight_layout()
    out = os.path.join(MODEL_DIR, "training_plot.png")
    plt.savefig(out, dpi=120)
    print(f"[PLOT] Saved to {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. EXPORT TO TFLITE
# ══════════════════════════════════════════════════════════════════════════════
def export_tflite(keras_model, out_path, quantize=True):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"[TFLITE] Exported to {out_path}  ({size_kb:.1f} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Driver Drowsiness — Level 2 Training")
    print("=" * 60)

    # ── load data ──────────────────────────────────────────────────────────
    X, y = load_dataset()
    if len(X) < 40:
        print("[ERROR] Not enough data. Run collect_data.py first.")
        print("        Need at least 20 awake + 20 drowsy images.")
        return

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    cw = class_weight.compute_class_weight("balanced",
                                           classes=np.unique(y_tr), y=y_tr)
    cw = dict(enumerate(cw))
    print(f"[DATA] Class weights: {cw}")

    # ── data augmentation ─────────────────────────────────────────────────
    aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomBrightness(0.15),
        layers.RandomContrast(0.15),
    ], name="augmentation")

    # ── PHASE 1: train CNN ────────────────────────────────────────────────
    print("\n[PHASE 1] Training CNN (frozen backbone)…")
    cnn, base = build_cnn()
    cnn.summary(line_length=70)

    cb_list = [
        callbacks.EarlyStopping(monitor="val_auc", patience=5,
                                restore_best_weights=True, mode="max"),
        callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5),
        callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "cnn_best.keras"),
                                  monitor="val_auc", save_best_only=True, mode="max"),
    ]

    def augment_ds(X_in, y_in, shuffle=True):
        ds = tf.data.Dataset.from_tensor_slices((X_in, y_in))
        if shuffle:
            ds = ds.shuffle(len(X_in))
        ds = ds.batch(BATCH_SIZE)
        if shuffle:
            ds = ds.map(lambda x, y: (aug(x, training=True), y),
                        num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)

    h1 = cnn.fit(
        augment_ds(X_tr, y_tr),
        validation_data=augment_ds(X_val, y_val, shuffle=False),
        epochs=EPOCHS_CNN,
        class_weight=cw,
        callbacks=cb_list,
        verbose=1,
    )

    # Fine-tune top layers of backbone
    print("\n[PHASE 1b] Fine-tuning top 20 backbone layers…")
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False
    cnn.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                loss="binary_crossentropy",
                metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    h1b = cnn.fit(
        augment_ds(X_tr, y_tr),
        validation_data=augment_ds(X_val, y_val, shuffle=False),
        epochs=10,
        class_weight=cw,
        callbacks=cb_list,
        verbose=1,
    )

    cnn_path = os.path.join(MODEL_DIR, "cnn_eye.keras")
    cnn.save(cnn_path)
    print(f"[SAVE] CNN saved to {cnn_path}")

    # CNN evaluation
    loss, acc, auc = cnn.evaluate(augment_ds(X_val, y_val, shuffle=False), verbose=0)
    print(f"[CNN] Val accuracy={acc*100:.1f}%  AUC={auc:.3f}")

    # ── PHASE 2: train CNN + LSTM ─────────────────────────────────────────
    print("\n[PHASE 2] Building sequences for LSTM…")
    feat_model = make_feature_model(cnn)
    feat_model.trainable = False

    Xs, ys = make_sequences(X, y)
    if len(Xs) < 10:
        print("[WARN] Too few sequences for LSTM training. Skipping LSTM phase.")
        print("       Collect more data (300+ samples per class) then re-run.")
    else:
        Xs_tr, Xs_val, ys_tr, ys_val = train_test_split(
            Xs, ys, test_size=0.2, stratify=ys, random_state=42)

        lstm_model = build_cnn_lstm(feat_model)
        lstm_model.summary(line_length=70)

        lstm_cb = [
            callbacks.EarlyStopping(monitor="val_auc", patience=5,
                                    restore_best_weights=True, mode="max"),
            callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "cnn_lstm_best.keras"),
                                      monitor="val_auc", save_best_only=True, mode="max"),
        ]

        h2 = lstm_model.fit(
            Xs_tr, ys_tr,
            validation_data=(Xs_val, ys_val),
            epochs=EPOCHS_LSTM,
            batch_size=16,
            callbacks=lstm_cb,
            verbose=1,
        )

        lstm_path = os.path.join(MODEL_DIR, "cnn_lstm.keras")
        lstm_model.save(lstm_path)
        print(f"[SAVE] LSTM model saved to {lstm_path}")

        loss, acc, auc = lstm_model.evaluate(Xs_val, ys_val, verbose=0)
        print(f"[LSTM] Val accuracy={acc*100:.1f}%  AUC={auc:.3f}")

    # ── PHASE 3: TFLite export ────────────────────────────────────────────
    print("\n[PHASE 3] Exporting to TFLite…")
    tflite_path = os.path.join(MODEL_DIR, "drowsiness.tflite")
    export_tflite(cnn, tflite_path, quantize=True)

    # Save metadata for inference
    meta = {
        "img_size":  IMG_SIZE,
        "seq_len":   SEQ_LEN,
        "classes":   CLASSES,
        "threshold": 0.5,
    }
    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[META] Saved model/meta.json")

    # Plot
    plot_history([h1], ["CNN"])
    print("\n[DONE] Training complete. Models saved in model/")


if __name__ == "__main__":
    main()
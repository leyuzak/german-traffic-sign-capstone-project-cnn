import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "traffic.keras"     
LABELS_PATH = APP_DIR / "labels.json"     
HISTORY_PATH = APP_DIR / "history.json"    

INPUT_SIZE = (30, 30)
NUM_CLASSES = 43


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(str(MODEL_PATH))


def load_labels():
    if LABELS_PATH.exists():
        try:
            labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
            if isinstance(labels, list) and len(labels) == NUM_CLASSES:
                return labels
        except Exception:
            pass
    return [f"Class {i}" for i in range(NUM_CLASSES)]


def preprocess(pil_img: Image.Image):
    img = np.array(pil_img.convert("RGB"))
    img = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # (1,30,30,3)
    return img


def topk(probs, k=5):
    idx = np.argsort(probs)[::-1][:k]
    return [(int(i), float(probs[i])) for i in idx]


st.set_page_config(page_title="Traffic Sign Classifier", layout="centered")
st.title("German Traffic Sign Classifier (GTSRB)")
st.write("Bir trafik işareti görseli yükleyin. Model Top-K tahmin döndürür.")

if not MODEL_PATH.exists():
    st.error(f"Model dosyası bulunamadı: {MODEL_PATH}\nRepo’ya traffic.keras ekle.")
    st.stop()

labels = load_labels()
model = load_model()

st.sidebar.header("Ayarlar")
k = st.sidebar.slider("Top-K", 1, 10, 5)
show_history = st.sidebar.checkbox("Training history (opsiyonel) göster", value=False)

file = st.file_uploader("Görsel yükle (jpg/png)", type=["jpg", "jpeg", "png"])
if file is None:
    st.info("Başlamak için bir görsel yükleyin.")
    st.stop()

pil_img = Image.open(file)
st.image(pil_img, use_container_width=True)

x = preprocess(pil_img)
probs = model.predict(x, verbose=0)[0]  # (43,)

st.subheader("Tahminler")
for cls_id, p in topk(probs, k=k):
    label_name = labels[cls_id] if cls_id < len(labels) else f"Class {cls_id}"
    st.write(f"**{label_name}** (id={cls_id}) — {p:.4f}")

if show_history:
    if HISTORY_PATH.exists():
        hist = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        st.subheader("Training History (history.json)")
        st.write({k: v[-1] if isinstance(v, list) and len(v) else v for k, v in hist.items()})
    else:
        st.warning("history.json bulunamadı (opsiyonel).")

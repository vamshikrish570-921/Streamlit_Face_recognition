"""
Face Scanner App — Streamlit + DeepFace
---------------------------------------
Features:
 - Capture a photo from your webcam (single-shot)
 - Upload an image manually
 - Analyze age and gender
 - Recognize people from a known-face database (add faces via sidebar)

Works with:
 Python 3.8
 TensorFlow 2.10.1 + Keras 2.10.0
 Streamlit 1.22.0
 DeepFace 0.0.93
"""

import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np
import os
import time
import pandas as pd

# ---------------- Configuration ----------------
KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

RECOG_MODEL = "Facenet"        # model for recognition
ANALYZE_ACTIONS = ["age", "gender"]
DETECTOR_BACKEND = "opencv"

st.set_page_config(page_title="Face Scanner App", layout="centered")
st.title("📷 Face Scanner — Age, Gender & Name Recognition")
st.write("Take a photo or upload one to detect age, gender, and recognize known faces.")

# ---------------- Utility functions ----------------
def save_uploaded_image(uploaded_file, dest_path):
    img = Image.open(uploaded_file).convert("RGB")
    img.save(dest_path, format="JPEG")
    return dest_path


def analyze_image(img_path):
    """Run DeepFace.analyze and return result dict."""
    try:
        obj = DeepFace.analyze(
            img_path=img_path,
            actions=ANALYZE_ACTIONS,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
        )
        if isinstance(obj, list):
            obj = obj[0]
        return obj
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return None


def find_best_match(img_path):
    """Search known faces directory and return best match (if any)."""
    try:
        df = DeepFace.find(
            img_path=img_path,
            db_path=KNOWN_FACES_DIR,
            model_name=RECOG_MODEL,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
        )
        if isinstance(df, list) and len(df) > 0:
            df = df[0]
        if isinstance(df, pd.DataFrame) and not df.empty:
            best = df.iloc[0]
            identity_path = best["identity"]
            distance_col = [
                c for c in df.columns if "VGG" in c or "distance" in c.lower()
            ]
            if distance_col:
                dist = best[distance_col[0]]
            else:
                dist = None
            name = os.path.splitext(os.path.basename(identity_path))[0]
            return {"name": name, "path": identity_path, "distance": dist}
        else:
            return None
    except Exception as e:
        st.warning(f"Recognition skipped: {e}")
        return None


def list_known_faces():
    return sorted(os.listdir(KNOWN_FACES_DIR))


# ---------------- Sidebar: Add Known Person ----------------
st.sidebar.header("Add a Known Person")
with st.sidebar.form("add_person"):
    new_name = st.text_input("Person's name")
    new_photo = st.file_uploader(
        "Upload a photo of this person", type=["jpg", "jpeg", "png"]
    )
    add_btn = st.form_submit_button("Add to Known Faces")

if add_btn:
    if not new_name or not new_photo:
        st.sidebar.error("Please enter a name and upload a photo.")
    else:
        safe_name = "".join(c for c in new_name if c.isalnum() or c in (" ", "_", "-"))
        filename = f"{safe_name.replace(' ', '_')}_{int(time.time())}.jpg"
        dest = os.path.join(KNOWN_FACES_DIR, filename)
        save_uploaded_image(new_photo, dest)
        st.sidebar.success(f"Added {new_name} to known faces ✅")

# ---------------- Main UI ----------------
mode = st.radio("Choose input mode:", ["Camera", "Upload Image", "Show Known Faces"])

if mode == "Show Known Faces":
    st.subheader("Known People")
    files = list_known_faces()
    if not files:
        st.info("No known people added yet.")
    else:
        cols = st.columns(3)
        for i, f in enumerate(files):
            with cols[i % 3]:
                st.image(os.path.join(KNOWN_FACES_DIR, f), caption=f, use_column_width=True)
else:
    if mode == "Camera":
        cam_file = st.camera_input("Take a photo and click Analyze")
        input_file = cam_file
    else:
        input_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if input_file is not None:
        image = Image.open(input_file).convert("RGB")
        st.image(image, caption="Input Image", use_column_width=True)

        temp_path = "temp_input.jpg"
        image.save(temp_path)

        st.markdown("---")
        st.header("Analysis")
        with st.spinner("Analyzing face..."):
            result = analyze_image(temp_path)

        if result:
            st.success("✅ Face analyzed successfully!")
            st.write(f"**Predicted Age:** {result.get('age')}")
            st.write(f"**Predicted Gender:** {result.get('gender')}")

            st.markdown("---")
            st.subheader("Searching known faces...")
            with st.spinner("Matching..."):
                match = find_best_match(temp_path)

            if match:
                st.success(f"Matched with: **{match['name']}** (distance: {match['distance']:.4f})")
                st.image(match["path"], caption=f"Matched: {match['name']}", use_column_width=True)
            else:
                st.info("No match found in known faces.")
        else:
            st.error("Could not analyze the image.")

# ---------------- Footer ----------------
st.markdown("---")
st.caption("💡 Tip: Add multiple images per person for better recognition accuracy.")

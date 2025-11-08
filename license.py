import streamlit as st
import torch
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# ---------------------------
# Title & Description
# ---------------------------
st.title("🚗 License Plate Recognition (Image & Video - CPU Version)")
st.write("Upload an image or a video to detect and read license plates using YOLO and EasyOCR.")

# ---------------------------
# Model & OCR Initialization
# ---------------------------
@st.cache_resource
def load_models():
    try:
        model = YOLO("yolov8n.pt")  # Small YOLO model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        model = None

    try:
        reader = easyocr.Reader(["en"], gpu=False)
    except Exception as e:
        st.error(f"Error loading OCR: {e}")
        reader = None

    return model, reader

yolo_model, reader = load_models()

# ---------------------------
# Helper Function for Image
# ---------------------------
def detect_license_plate_image(image, model, reader):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = model(img_cv)
    annotated_img = img_cv.copy()
    plates_text = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        plate_crop = img_cv[y1:y2, x1:x2]
        if plate_crop.size == 0:
            continue
        ocr_result = reader.readtext(plate_crop)
        for (_, text, conf_text) in ocr_result:
            if conf_text > 0.3:
                clean_text = text.strip().upper()
                plates_text.append(clean_text)
                cv2.putText(annotated_img, clean_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    unique_texts = list(dict.fromkeys(plates_text))
    return annotated_img, unique_texts

# ---------------------------
# Helper Function for Video
# ---------------------------
def detect_license_plate_video(video_path, model, reader):
    cap = cv2.VideoCapture(video_path)
    plates_text = []
    frame_count = 0

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = model(frame)
        annotated_frame = frame.copy()

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            plate_crop = frame[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue
            ocr_result = reader.readtext(plate_crop)
            for (_, text, conf_text) in ocr_result:
                if conf_text > 0.3:
                    clean_text = text.strip().upper()
                    plates_text.append(clean_text)
                    cv2.putText(annotated_frame, clean_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Convert to RGB for Streamlit display
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        stframe.image(annotated_frame_rgb, channels="RGB", use_container_width=True)

    cap.release()
    unique_texts = list(dict.fromkeys(plates_text))
    return unique_texts

# ---------------------------
# File Upload UI
# ---------------------------
uploaded_file = st.file_uploader("📤 Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    if file_extension in ["jpg", "jpeg", "png"]:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        st.write("🔍 Running License Plate Detection...")
        annotated_img, unique_texts = detect_license_plate_image(image, yolo_model, reader)

        st.image(annotated_img, caption="Detected License Plate", use_container_width=True)
        if unique_texts:
            st.success("✅ Detected License Plate Text:")
            for t in unique_texts:
                st.write(f"📄 {t}")
        else:
            st.warning("No readable license plate text detected.")

    elif file_extension in ["mp4", "avi", "mov"]:
        st.video(uploaded_file)
        st.write("🎞️ Processing video frames for license plate detection...")

        unique_texts = detect_license_plate_video(tmp_path, yolo_model, reader)
        if unique_texts:
            st.success("✅ Detected License Plate Texts from Video:")
            for t in unique_texts:
                st.write(f"📄 {t}")
        else:
            st.warning("No readable license plate text detected in video.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("**Developed by:** BATCH-6 III AIML-A")
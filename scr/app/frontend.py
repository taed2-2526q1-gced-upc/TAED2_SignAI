"""
Streamlit app for SignAI - Traffic Sign Detection

This UI sends an image to the FastAPI backend and displays predictions.
Run with: streamlit run scr/app/frontend.py
"""

import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io
import pandas as pd

# Backend API URL (adjust if needed)
API_URL = "http://127.0.0.1:8080/predict"

st.set_page_config(page_title="SignAI - Traffic Sign Detection", page_icon="🚦")
st.title("🚗 SignAI - Traffic Sign Detection")
st.markdown("Upload an image of a road scene and detect traffic signs using your YOLOv8 model.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Detection"):
        with st.spinner("Detecting traffic signs..."):
            # Send image to FastAPI backend
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            buf.seek(0)
            files = {"file": ("image.jpg", buf, "image/jpeg")}
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                data = response.json()
                preds = data.get("predictions", [])

                if not preds:
                    st.warning("⚠️ No signs detected in the image.")
                else:
                    # Draw bounding boxes and labels
                    draw = ImageDraw.Draw(image)

                    for p in preds:
                        bbox = p["bbox"]
                        label = f"{p['class']} ({p['confidence']:.2f})"
                        draw.rectangle(bbox, outline="red", width=3)
                        draw.text((bbox[0], bbox[1] - 15), label, fill="red")

                    # Show the annotated image
                    st.image(image, caption="Detected Signs", use_container_width=True)
                    st.success(f"✅ Detected {len(preds)} sign(s).")

                    # ✅ Show a table of detections
                    df = pd.DataFrame(preds)
                    st.markdown("### 🧾 Detection results")
                    st.dataframe(df, use_container_width=True)

            else:
                st.error(f"API returned error {response.status_code}: {response.text}")

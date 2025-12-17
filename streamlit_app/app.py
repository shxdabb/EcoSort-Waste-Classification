import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# ---- Page Settings ----
st.set_page_config(page_title="Waste Classifier", layout="wide")

# ---- Load Model ----
model = load_model("model/waste_classifier_model.h5")

# Must match training order
classes = ['battery','biological','cardboard','clothes','glass',
           'metal','paper','plastic','shoes','trash']

# ---- Title ----
st.title("♻️ Waste Classification System")

st.write("Upload an image to classify it into one of the 10 waste categories.")

# ---- Layout (Side-by-Side) ----
col1, col2 = st.columns(2)

with col1:
    uploaded = st.file_uploader("Choose a waste image", type=['jpg','jpeg','png'])

    if uploaded:
        # Read image
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Display Image
        st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

        # Preprocess
        resized = cv2.resize(img_rgb, (64,64))
        normalized = resized / 255.0
        input_img = np.expand_dims(normalized, axis=0)

        # Predict
        preds = model.predict(input_img)
        class_id = np.argmax(preds)
        confidence = np.max(preds) * 100
        result = classes[class_id]

with col2:
    st.subheader("Prediction")

    if uploaded:
        st.markdown(f"### Predicted Category: **{result.upper()}**")
        st.write(f"**Confidence:** {confidence:.2f}%")
    else:
        st.info("Upload an image to see the classification result here.")

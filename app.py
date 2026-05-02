import streamlit as st

st.write("ISOM5240")
st.write("ISOM5240")
st.write("ISOM5240")
st.write("ISOM5240")

st.title("Title")
st.header("Header")
st.subheader("Sub-header")

st.write("Hello, *World!* :sunglasses:")


import streamlit as st
from transformers import pipeline
from PIL import Image

# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(
    page_title="Gender Classification App",
    page_icon="🧑",
    layout="centered"
)

st.title("🧑 Gender Classification using ViT")
st.write("Upload a human image, and the model will classify it into male/female categories.")

# -----------------------------
# Load model only once
# -----------------------------
@st.cache_resource
def load_gender_classifier():
    return pipeline(
        "image-classification",
        model="rizvandwiki/gender-classification"
    )

gender_classifier = load_gender_classifier()

# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Classifying gender..."):
        gender_predictions = gender_classifier(image)

    gender_predictions = sorted(
        gender_predictions,
        key=lambda x: x["score"],
        reverse=True
    )

    top_prediction = gender_predictions[0]

    st.subheader("Predicted Gender Category")
    st.success(f"Prediction: {top_prediction['label']}")

    st.subheader("Prediction Details")

    for prediction in gender_predictions:
        label = prediction["label"]
        score = prediction["score"]

        st.write(f"{label}: {score:.2%}")
        st.progress(score)

else:
    st.info("Please upload an image to start gender classification.")

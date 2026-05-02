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
    page_title="Age Classification App",
    page_icon="🧑",
    layout="centered"
)

st.title("🧑 Age Classification using ViT")
st.write("Upload a face image, and the model will predict the person's age range.")

# -----------------------------
# Load model only once
# -----------------------------
@st.cache_resource
def load_age_classifier():
    return pipeline(
        "image-classification",
        model="nateraw/vit-age-classifier"
    )

age_classifier = load_age_classifier()

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

    with st.spinner("Classifying age..."):
        age_predictions = age_classifier(image)

    age_predictions = sorted(
        age_predictions,
        key=lambda x: x["score"],
        reverse=True
    )

    top_prediction = age_predictions[0]

    st.subheader("Predicted Age Range")
    st.success(f"Age range: {top_prediction['label']}")

    st.subheader("Prediction Details")

    for prediction in age_predictions:
        label = prediction["label"]
        score = prediction["score"]

        st.write(f"{label}: {score:.2%}")
        st.progress(score)

else:
    st.info("Please upload an image to start age classification.")

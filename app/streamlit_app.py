import os
from pathlib import Path

import streamlit as st
from PIL import Image

from src.predict import FashionClassifier

st.set_page_config(page_title="Fashion Classifier", page_icon="🛍️")
st.title("Fashion Product Classifier")
model_path = Path(os.getenv("MODEL_PATH", "models/best_model.pth"))


@st.cache_resource
def load_classifier(path: str) -> FashionClassifier:
    return FashionClassifier(path)


if not model_path.is_file():
    st.error(f"Model checkpoint not found: {model_path}. Train a model first.")
else:
    classifier = load_classifier(str(model_path))
    uploaded_file = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            result = classifier.predict(image)
        except (OSError, Image.DecompressionBombError) as error:
            st.error(f"Unable to read image: {error}")
        else:
            left, right = st.columns(2)
            left.image(image, use_container_width=True)
            right.metric("Category", result["category"])
            right.metric("Confidence", f"{result['confidence']:.1%}")
            for prediction in result["top_k"]:
                right.progress(
                    prediction["score"],
                    text=f"{prediction['label']} — {prediction['score']:.1%}",
                )

import numpy as np
import pandas as pd
from pathlib import Path
import io
from PIL import Image
import streamlit as st
import tensorflow as tf

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMG_SIZE = (32, 32)
MODEL_CANDIDATES = [Path("model_retrained.h5"), Path("model.h5")]


@st.cache_resource
def load_model() -> tf.keras.Model:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return tf.keras.models.load_model(path)
    raise FileNotFoundError("No model weights found. Expected one of: model_retrained.h5 or model.h5 in project root.")


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize, force RGB, and normalize to match training pipeline."""
    image = image.convert("RGB").resize(IMG_SIZE)
    return np.asarray(image, dtype=np.float32) / 255.0


def predict(image: Image.Image):
    model = load_model()
    arr = preprocess_image(image)
    preds = model.predict(arr[np.newaxis, ...], verbose=0)[0]
    top_idx = int(np.argmax(preds))
    return CLASS_NAMES[top_idx], float(preds[top_idx]), preds


def get_model_metadata(model: tf.keras.Model) -> dict:
    buffer = io.StringIO()
    model.summary(print_fn=lambda x: buffer.write(x + "\n"))
    return {
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "parameters": int(model.count_params()),
        "summary": buffer.getvalue(),
        "weights_path": Path("model.h5").resolve(),
    }


def get_dataset_stats(root: Path) -> pd.DataFrame:
    counts = []
    for cls in CLASS_NAMES:
        cls_dir = root / cls
        if cls_dir.exists():
            counts.append({"class": cls, "images": len(list(cls_dir.glob("*.jpg")))})
    return pd.DataFrame(counts)


def main():
    st.set_page_config(page_title="Garbage Classifier", page_icon="🗑️")
    st.title("Garbage Classification (6 classes)")
    st.caption("Upload a photo of cardboard, glass, metal, paper, plastic, or trash. "
               "We resize to 32×32 RGB and normalize for you.")

    st.sidebar.header("How to use")
    st.sidebar.markdown(
        "- Upload a clear object photo (one item, centered)\n"
        "- Neutral background helps\n"
        "- If unsure, inspect the debug view"
    )
    top_k = st.sidebar.slider("Top predictions to show", min_value=3, max_value=len(CLASS_NAMES), value=6)
    st.sidebar.divider()
    st.sidebar.write("Weights checked in order:")
    for path in MODEL_CANDIDATES:
        st.sidebar.write(f"• {path.name}")

    with st.expander("Model & dataset details"):
        try:
            model = load_model()
            meta = get_model_metadata(model)
            st.markdown(
                f"- Input shape: `{meta['input_shape']}`\n"
                f"- Output shape: `{meta['output_shape']}`\n"
                f"- Parameters: `{meta['parameters']:,}`\n"
                f"- Weights file: `{meta['weights_path']}`\n"
                f"- Classes (order-sensitive): `{', '.join(CLASS_NAMES)}`\n"
                f"- Preprocessing: resize to {IMG_SIZE[0]}×{IMG_SIZE[1]}, force RGB, scale 0–1"
            )
            st.code(meta["summary"], language="text")
        except Exception as err:
            st.warning(f"Could not load model metadata: {err}")

        stats = get_dataset_stats(Path("Garbage/processed_images"))
        if not stats.empty:
            st.write("Class balance (processed_images):")
            st.dataframe(stats, hide_index=True)
        else:
            st.info("Dataset stats unavailable (Garbage/processed_images missing?).")

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded image", use_container_width=True)

        label, score, probs = predict(image)
        st.success(f"Prediction: {label} — {score * 100:.1f}% confidence")

        prob_df = pd.DataFrame({"class": CLASS_NAMES, "confidence": probs * 100})

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top classes")
            st.dataframe(
                prob_df.sort_values("confidence", ascending=False).head(top_k).reset_index(drop=True),
                hide_index=True,
                use_container_width=True,
            )
        with col2:
            st.subheader("Confidence chart")
            st.bar_chart(prob_df.set_index("class"))

        with st.expander("Debug view (model input)"):
            resized = image.convert("RGB").resize(IMG_SIZE)
            st.image(resized, caption=f"Resized to {IMG_SIZE[0]}×{IMG_SIZE[1]}", width=200)
    else:
        st.info("Upload an image to see predictions.")


if __name__ == "__main__":
    main()

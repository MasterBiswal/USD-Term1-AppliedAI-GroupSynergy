import streamlit as st
import joblib
import numpy as np
import os

# === Load model and reduced features shape ===
model = joblib.load("saved_model/xgb_multilabel_model.pkl")
X_reduced = joblib.load("artifacts/X_reduced.pkl")

# === Load human-readable tag names ===
tag_classes_path = "data/processed/tag_classes.txt"
if os.path.exists(tag_classes_path):
    with open(tag_classes_path, "r") as f:
        tag_classes = [line.strip() for line in f.readlines()]
else:
    tag_classes = [f"Tag {i}" for i in range(model.n_classes_)]  # fallback

# === Streamlit App Setup ===
st.set_page_config(page_title="🔖 Tag Predictor", layout="wide")
st.title("🔖 Multi-label Tag Predictor")
st.markdown("Use feature values to predict relevant tags. You can also choose a sample from existing data.")

# === Initialize session state ===
if "selected_sample_index" not in st.session_state:
    st.session_state.selected_sample_index = 0

if "example_vector" not in st.session_state:
    st.session_state.example_vector = [0.0] * X_reduced.shape[1]

# === Dropdown to Select Sample Index ===
st.subheader("🎯 Load a Pre-filled Example")
sample_index = st.selectbox(
    "Choose a sample index from dataset (0 to {})".format(len(X_reduced) - 1),
    options=list(range(min(100, len(X_reduced)))),  # limit to 100 for UI speed
    index=st.session_state.selected_sample_index,
)

# === Update sample on dropdown change ===
if sample_index != st.session_state.selected_sample_index:
    st.session_state.selected_sample_index = sample_index
    st.session_state.example_vector = X_reduced[sample_index].tolist()
    st.success(f"✅ Sample {sample_index} loaded!")

# === Create Form Inputs Based on Feature Length ===
num_features = X_reduced.shape[1]
input_data = []

with st.form("prediction_form"):
    st.subheader("🧮 Input Feature Values")
    cols = st.columns(3)

    for i in range(num_features):
        default_val = st.session_state.example_vector[i]
        val = cols[i % 3].number_input(f"Feature {i+1}", value=default_val, step=0.1, format="%.3f")
        input_data.append(val)

    submitted = st.form_submit_button("🎯 Predict Tags")

# === Prediction & Output ===
if submitted:
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    predicted_indices = np.where(prediction[0] == 1)[0]

    if len(predicted_indices) == 0:
        st.warning("⚠️ No tags predicted for the given input.")
    else:
        st.success("✅ Predicted Tags:")
        for i in predicted_indices:
            st.markdown(f"- {tag_classes[i]}")

# app.py
import sys
import os
import streamlit as st
from PIL import Image
import io

# --- Configuration ---
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
if backend_path not in sys.path:
    sys.path.append(backend_path)

# --- Import Backend Function ---
try:
    # Import the function that returns prediction and heatmap
    from backend.inference import visualize_attention, device
except ImportError as e:
    st.error(f"Error importing backend module: {e}. Please ensure 'backend/inference.py' and 'backend/levit_model.py' exist and are correctly structured.")
    st.stop()
except Exception as e: 
    st.error(f"An unexpected error occurred while initializing the backend: {e}")
    # Optionally print traceback to console for debugging
    # import traceback
    # traceback.print_exc()
    st.stop()


# --- Streamlit App Layout ---
st.set_page_config(page_title="Pneumonia Detector", layout="wide", initial_sidebar_state="expanded")

# Sidebar for information
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "**Pneumonia Detection AI**\n\n"
    "Uses a LeViT model to predict Pneumonia from chest X-rays.\n\n"
    "**How it works:**\n"
    "1. Upload an X-ray.\n"
    "2. The model analyzes it.\n"
    "3. View prediction and attention heatmap.\n\n" # Updated text
    f"**Model running on:** `{str(device).upper()}`"
)
st.sidebar.markdown("---")
st.sidebar.warning("**Disclaimer:** For informational purposes only. Not medical advice.")


# Main page content
st.title("🩺 Pneumonia Detection from Chest X-rays")
st.markdown("Upload an image. The AI predicts Normal/Pneumonia and shows an **Attention Heatmap** indicating areas the model focused on.") # Updated description

uploaded_file = st.file_uploader("⬆️ Choose a chest X-ray image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.stop()

    # --- Display Original Image ---
    st.markdown("---")
    st.subheader("🖼️ Uploaded X-ray")
    st.image(image, caption="Your Uploaded Image", use_column_width=True)

    # --- Prediction & Visualization ---
    temp_image_path = f"temp_{uploaded_file.name}"
    try:
        image.save(temp_image_path)
    except Exception as e:
        st.error(f"Error saving temporary file: {e}")
        st.stop()

    prediction = None
    confidence = None
    heatmap_image = None # Variable to hold the result

    with st.spinner("🧠 Analyzing image and generating heatmap... Please wait."):
        try:
            # Call the backend function
            prediction, confidence, heatmap_image = visualize_attention(temp_image_path)
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            # import traceback # Optional: log traceback server-side
            # traceback.print_exc()
        finally:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

    st.markdown("---")
    st.subheader("🔍 Analysis Results")

    # Check if prediction and heatmap were successful
    if prediction is not None and confidence is not None:
        if prediction == "Pneumonia":
            st.error(f"**Prediction:** {prediction} (Confidence: {confidence:.2%})")
        else:
            st.success(f"**Prediction:** {prediction} (Confidence: {confidence:.2%})")

        # Display Heatmap if available
        if heatmap_image is not None and isinstance(heatmap_image, Image.Image):
            st.subheader("🔥 Attention Heatmap")
            st.image(heatmap_image, caption="Attention heatmap overlay (Areas influencing the prediction)", use_column_width=True)
            st.caption("Brighter areas indicate regions the model paid more attention to.")
        else:
             st.warning("Could not generate attention heatmap for this image.")

    else:
        # Handle cases where prediction failed entirely
        st.error("Could not analyze the image. Please ensure it's a valid chest X-ray image and try again.")

else:
    st.info("Please upload an image to start the analysis.")

# Optional: Add a footer
st.markdown("---")
st.markdown("Developed with Streamlit and PyTorch. Model: LeViT.")
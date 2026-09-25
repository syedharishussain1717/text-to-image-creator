
"""
AI Text-to-Image Generator

A beginner-friendly Streamlit app that turns text prompts into images
using Hugging Face Inference Providers.
"""

import io
from PIL import Image
import streamlit as st
from huggingface_hub import InferenceClient


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="centered"
)


# ---------------------------------------------------------
# Hugging Face settings
# ---------------------------------------------------------

# Read the Hugging Face token from Streamlit Secrets.
# Never put the real token directly in this file.
HF_TOKEN = st.secrets.get("HF_TOKEN", "")


# ---------------------------------------------------------
# Image generation function
# ---------------------------------------------------------
def generate_image(prompt: str, num_steps: int, guidance_scale: float):
    """
    Sends the prompt to Hugging Face and returns
    the generated image as a PIL Image.
    """

    client = InferenceClient(
        api_key=HF_TOKEN
    )

    image = client.text_to_image(
        prompt=prompt,
        model="black-forest-labs/FLUX.1-schnell",
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale
    )

    return image


# ---------------------------------------------------------
# Sidebar — generation settings
# ---------------------------------------------------------
st.sidebar.title("⚙️ Settings")

num_steps = st.sidebar.slider(
    "Quality (inference steps)",
    min_value=10,
    max_value=50,
    value=25,
    step=5,
    help="More steps can improve image quality but may take longer."
)

guidance_scale = st.sidebar.slider(
    "Prompt strength (guidance scale)",
    min_value=1.0,
    max_value=15.0,
    value=7.5,
    step=0.5,
    help="Higher values make the generated image follow the prompt more strictly."
)


# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------
st.title("🎨 AI Text-to-Image Generator")

st.write(
    "Type a description below and let AI turn it into an image."
)


# ---------------------------------------------------------
# Prompt input
# ---------------------------------------------------------
prompt = st.text_input(
    "Enter your image prompt",
    placeholder="e.g. a cute corgi wearing sunglasses, digital art"
)


# ---------------------------------------------------------
# Generate button
# ---------------------------------------------------------
generate_btn = st.button(
    "Generate Image",
    type="primary"
)


# ---------------------------------------------------------
# Image display area
# ---------------------------------------------------------
image_area = st.empty()


# ---------------------------------------------------------
# Generate image
# ---------------------------------------------------------
if generate_btn:

    # Check whether the user entered a prompt
    if not prompt.strip():

        st.warning("Please enter a prompt first.")

    # Check whether the Hugging Face token exists
    elif not HF_TOKEN:

        st.error(
            "No Hugging Face token found. "
            "Add HF_TOKEN under your Streamlit app's "
            "Settings → Secrets."
        )

    else:

        with st.spinner(
            "Generating your image... this may take a moment ⏳"
        ):

            try:

                image = generate_image(
                    prompt,
                    num_steps,
                    guidance_scale
                )

            except Exception as e:

                st.error(
                    f"Image generation failed: {e}"
                )

                image = None


        # -------------------------------------------------
        # Display generated image
        # -------------------------------------------------
        if image is not None:

            image_area.image(
                image,
                caption=prompt,
                use_container_width=True
            )


            # -------------------------------------------------
            # Convert image to bytes for downloading
            # -------------------------------------------------
            img_bytes = io.BytesIO()

            image.save(
                img_bytes,
                format="PNG"
            )

            img_bytes.seek(0)


            # -------------------------------------------------
            # Download button
            # -------------------------------------------------
            st.download_button(
                label="⬇️ Download Image",
                data=img_bytes,
                file_name="generated_image.png",
                mime="image/png"
            )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.caption(
    "Powered by FLUX.1-schnell · Hugging Face · Streamlit"
)


"""
AI Text-to-Image Generator
A beginner-friendly Streamlit app that turns text prompts into images
using the Stable Diffusion v1.5 model, generated via Hugging Face's
hosted Inference API (so no GPU or heavy libraries are needed locally
or on Streamlit Cloud).
"""

import io
import requests
from PIL import Image
import streamlit as st

# ---------------------------------------------------------
# Page setup — this must be the first Streamlit command
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="centered"
)

# ---------------------------------------------------------
# Hugging Face Inference API settings
# ---------------------------------------------------------
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

# The token is read from Streamlit "Secrets" (never hard-code it in code)
HF_TOKEN = st.secrets.get("HF_TOKEN", "")


def generate_image(prompt: str, num_steps: int, guidance_scale: float):
    """
    Sends the prompt to Hugging Face's Inference API and returns
    a PIL Image. Raises an error with a readable message if it fails.
    """
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "num_inference_steps": num_steps,
            "guidance_scale": guidance_scale
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)

    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))

    # The API returns JSON (not an image) when something goes wrong
    try:
        error_msg = response.json().get("error", response.text)
    except ValueError:
        error_msg = response.text

    # Common case: model is "cold" and needs to spin up on HF's servers
    if "loading" in error_msg.lower():
        raise RuntimeError(
            "The model is warming up on Hugging Face's servers. "
            "This can take 20-60 seconds the first time — please try again shortly."
        )

    raise RuntimeError(f"Image generation failed: {error_msg}")


# ---------------------------------------------------------
# Sidebar — simple settings so the user can tweak generation
# ---------------------------------------------------------
st.sidebar.title("⚙️ Settings")

num_steps = st.sidebar.slider(
    "Quality (inference steps)",
    min_value=10, max_value=50, value=25, step=5,
    help="More steps = better quality, but slower generation."
)

guidance_scale = st.sidebar.slider(
    "Prompt strength (guidance scale)",
    min_value=1.0, max_value=15.0, value=7.5, step=0.5,
    help="Higher = follows your prompt more strictly."
)


# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------
st.title("🎨 AI Text-to-Image Generator")
st.write("Type a description below and let AI turn it into an image.")

prompt = st.text_input(
    "Enter your image prompt",
    placeholder="e.g. a cute corgi wearing sunglasses, digital art"
)

generate_btn = st.button("Generate Image", type="primary")

# Placeholder area where the image will appear
image_area = st.empty()

if generate_btn:
    if not prompt.strip():
        st.warning("Please enter a prompt first.")
    elif not HF_TOKEN:
        st.error(
            "No Hugging Face token found. Add one under your app's "
            "Settings → Secrets as HF_TOKEN (see README for steps)."
        )
    else:
        with st.spinner("Generating your image... this may take a moment ⏳"):
            try:
                image = generate_image(prompt, num_steps, guidance_scale)
            except RuntimeError as e:
                st.error(str(e))
                image = None

        if image is not None:
            # Show the generated image
            image_area.image(image, caption=prompt, use_container_width=True)

            # Convert image to bytes so it can be downloaded
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            st.download_button(
                label="⬇️ Download Image",
                data=img_bytes,
                file_name="generated_image.png",
                mime="image/png"
            )

st.caption("Powered by Stable Diffusion v1.5 · Hugging Face · Streamlit")

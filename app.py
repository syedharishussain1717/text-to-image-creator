"""
AI Text-to-Image Generator
A beginner-friendly Streamlit app that turns text prompts into images
using the Stable Diffusion v1.5 model from Hugging Face.
"""

import io
import torch
import streamlit as st
from diffusers import StableDiffusionPipeline

# ---------------------------------------------------------
# Page setup — this must be the first Streamlit command
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="centered"
)


# ---------------------------------------------------------
# Load the model once and cache it
# (st.cache_resource keeps it in memory instead of reloading
# every time the user interacts with the app)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype
    )
    pipe = pipe.to(device)

    return pipe, device


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
    else:
        with st.spinner("Generating your image... this may take a moment ⏳"):
            pipe, device = load_model()

            if device == "cpu":
                st.info("Running on CPU — generation will be slower than on a GPU.")

            result = pipe(
                prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale
            )
            image = result.images[0]

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

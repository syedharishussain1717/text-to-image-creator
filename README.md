# AI Text-to-Image Generator

A simple Streamlit web app that generates images from text prompts using
Stable Diffusion v1.5 (Hugging Face).

## Project Structure

```
text2image-app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Run Locally

1. Create a virtual environment (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the app:
   ```
   streamlit run app.py
   ```

4. It will open automatically in your browser at `http://localhost:8501`

## Notes

- If you have an NVIDIA GPU with CUDA installed, the app will automatically
  use it (much faster). Otherwise it falls back to CPU, which is slower
  (can take a few minutes per image).
- The first run will download the model (~4-5 GB), so it needs a good
  internet connection and some patience the first time.

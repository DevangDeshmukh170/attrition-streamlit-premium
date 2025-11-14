# Attrition Streamlit Premium App

This repository contains a premium Streamlit app for Employee Attrition prediction.
It includes a pre-trained model artifact (`attrition_model_pipeline.pkl`), a multi-page Streamlit app,
and instructions for running locally or deploying on Streamlit Cloud / Docker.

## Files
- `app.py` : Streamlit application (multi-page)
- `requirements.txt` : Python dependencies
- `attrition_model_pipeline.pkl` : Pre-trained model + scaler + columns (included)

## Run locally
1. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run Streamlit:
   ```bash
   streamlit run app.py
   ```

## Deploy
- Push this repo to GitHub and deploy on Streamlit Community Cloud (share.streamlit.io)
- Or build Docker image using provided Dockerfile (optional) and run.

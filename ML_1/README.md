# Student Depression Risk Screener (Streamlit)

## Run locally
```bash
pip install -r requirements.txt
python train_model.py Student_Depression_Dataset.csv   # creates model.joblib (run once)
streamlit run app.py
``
#step1
Put `Student_Depression_Dataset.csv` in the same folder as `train_model.py`.

## Deploy free on Streamlit Community Cloud
1. Create a GitHub repo containing: `app.py`, `train_model.py`, `requirements.txt`, `model.joblib`
   (do NOT upload the dataset if you are unsure about its licence).
2. Go to share.streamlit.io, connect the repo, and set the main file to `app.py`.

## Files
- `train_model.py`: trains Logistic Regression and saves `model.joblib` plus metadata for the app
- `app.py`: the web interface (input form, score, explanation chart, model info tab)

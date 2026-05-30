# Deploying to Streamlit Community Cloud (free)

The Streamlit app is **self-contained**: if no API is reachable it loads the model
in-process, so it deploys as a single app with nothing else to host.

## Prerequisites

- `models/price_classifier.pkl` is committed (it is — required so the cloud host has
  the model).
- The repo is pushed to GitHub.

## Steps

1. **Push to GitHub:**
   ```bash
   git remote add origin https://github.com/<you>/makanpredict-api.git
   git push -u origin main
   ```
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. **Create app** → pick your repo, branch **`main`**, main file **`streamlit_app.py`**.
4. *(Optional)* Advanced settings → Python **3.11** or **3.12** (best wheel coverage for
   the pinned `scikit-learn` / `xgboost`).
5. **Deploy.** You'll get a public URL like `https://<app>.streamlit.app` — share that.

## Notes

- `requirements.txt` is installed automatically. `scikit-learn` and `xgboost` are pinned
  to the versions that trained the `.pkl`, so unpickling matches.
- First load takes a few seconds (model load); it's cached afterwards.
- The free tier has limited RAM — the model + pandas/xgboost fit comfortably.
- **Want the live site to call a real API instead of loading in-process?** Host the
  FastAPI service somewhere (Render, Hugging Face, Fly.io) and set a Streamlit secret /
  env var `MAKANPREDICT_API = https://your-api-host`. The UI will use it automatically.

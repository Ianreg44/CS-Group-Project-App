# Smart Cookbook — Streamlit deployment files

Use these files in your GitHub repo root:

- `streamlit_app.py` — main Streamlit app file
- `requirements.txt` — tells Streamlit Cloud to install all needed Python libraries
- `.gitignore` — prevents local cache/database/secrets from being committed

## Deploy checklist

1. Put `streamlit_app.py`, `requirements.txt`, and `.gitignore` at the root of your GitHub repository.
2. Commit and push to GitHub.
3. In Streamlit Cloud, set the main file path to `streamlit_app.py`.
4. Reboot/redeploy the app.

The error `ModuleNotFoundError: No module named 'plotly'` is fixed by having `plotly` in `requirements.txt`.

# TSP Adventure Game (Streamlit)

An interactive Travelling Salesperson Game with BFS/DFS/UCS/A* comparisons.

## Local run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy to Streamlit Community Cloud

1. Push this folder to a public GitHub repo.
2. Go to https://share.streamlit.io (Community Cloud dashboard).
3. Click **Create app** → select your repo/branch → set **file path** to `streamlit_app.py`.
4. (Optional) In **Advanced settings**, choose the Python version (e.g., 3.11) if your libs need it.
5. Click **Deploy**. Your app will be live at `https://<your-subdomain>.streamlit.app`.


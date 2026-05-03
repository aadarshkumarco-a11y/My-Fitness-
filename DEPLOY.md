# Deploying to Streamlit Community Cloud

This repo is pre-configured for one-click deployment on
[Streamlit Community Cloud](https://share.streamlit.io). Follow the steps below.

## 1. Push this folder to GitHub

```bash
cd nse-backtester-repo
git init -b main
git add .
git commit -m "Initial commit: NSE backtesting platform"
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

A **public** repo is the simplest path. Private repos also work but require
granting Streamlit Cloud access to your GitHub account.

## 2. Connect Streamlit Cloud

1. Go to <https://share.streamlit.io>.
2. Sign in with the same GitHub account.
3. Click **Create app** → **Deploy from existing repo**.
4. Fill in:
   - **Repository:** `<your-username>/<repo-name>`
   - **Branch:** `main`
   - **Main file path:** `nse_backtester/ui/app.py`
   - **App URL (optional):** pick a subdomain like `nse-backtester`.
5. Click **Deploy**.

The first build takes ~2-3 minutes. Streamlit will install everything in
`requirements.txt` automatically. Subsequent deploys (every push to `main`)
take seconds.

## 3. Verify

When deployment finishes you'll get a URL like
`https://<your-app>.streamlit.app`. Open it and:

* Pick a strategy in the sidebar (try `ema_crossover`).
* Click **▶ Run backtest**.
* You should see KPI cards, equity curve, drawdown chart, and a trade table.

## Notes & gotchas

* **NSE live scraping won't work from Streamlit Cloud's IPs** — NSE blocks
  most cloud datacenters. The "Synthetic" data source always works. To use
  live data, either:
  - Run the app locally where your residential IP is allowed.
  - Add a paid residential proxy via env vars (advanced).
* **State persistence**: Streamlit Cloud's filesystem is *ephemeral*. The
  SQLite DB and CSV exports written under `nse_backtester/data/` will be
  wiped on every restart. For permanent storage, use a mounted volume on
  Render/Railway/Fly.io instead, or wire up an external Postgres.
* **Memory**: free tier has ~1 GB. Synthetic backtests on 365 days are well
  under 100 MB. Don't crank `days` past ~5000 on the free tier.

## Updating the app

```bash
git add .
git commit -m "Tweak strategy params"
git push
```

Streamlit Cloud auto-redeploys on push.

## Local development (no cloud needed)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run nse_backtester/ui/app.py
pytest -q nse_backtester/tests
```

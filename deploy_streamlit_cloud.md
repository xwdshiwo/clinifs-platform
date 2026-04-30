# Deploy to Streamlit Community Cloud

## Prerequisites

- A GitHub account
- A public GitHub repository containing the `platform/` directory

---

## Step 1 — Push platform to GitHub

Option A: same repository as the package  
Option B: a dedicated repository `clinifs-platform` (recommended for clean separation)

```bash
# Example: create a new repo and push platform/
cd 2026-04-30_1100_final_submission_package/platform
git init
git add .
git commit -m "Initial clinifs web platform (v0.1.0)"
git remote add origin https://github.com/<user>/clinifs-platform.git
git push -u origin main
```

---

## Step 2 — Create the Streamlit Community Cloud app

1. Go to **https://share.streamlit.io** and sign in with your GitHub account.
2. Click **"New app"**.
3. Select the repository: `<user>/clinifs-platform`
4. Set **Branch**: `main`
5. Set **Main file path**: `app/main.py`
6. (Optional) Set **App URL**: request `clinifs` as subdomain → `https://clinifs.streamlit.app`
7. Click **"Deploy!"**

---

## Step 3 — Wait for build & check logs

The build installs `requirements.txt` automatically. Expected build time: 3–5 minutes.

If the build fails, check logs for:
- Missing packages → add to `requirements.txt`
- Import errors → verify `sys.path` insertion in page files (already handled)

---

## Step 4 — Smoke test each page

| Page | Test |
|---|---|
| Run Analysis | Upload `examples/example_microarray.csv` + `examples/example_labels.csv`; select "auto" and k=20; click Run |
| Browse Results | Filter by "Hard" tier; verify 2 datasets appear |
| Get Recommendation | Select "Hard" + "Predictive performance"; verify `RRA(ANOVA + MI + MEL)` is recommended |
| Custom RRA | Upload example CSV; run RRA(ANOVA+MI); verify ρ-score distribution chart appears |

---

## Step 5 — Record URL and write into paper

After deployment, record:
- **App URL**: e.g. `https://clinifs.streamlit.app`
- **Commit hash**: `git rev-parse --short HEAD`

Write both into:
- `paper/main_v5_zh.md` → §"数据与代码可用性"
- `package/README.md` → "Quick links" table

---

## Re-deploy after updates

Push new commits to the GitHub repository. Streamlit Cloud automatically re-deploys on each push to `main`.

To pin a specific version, tag the commit:

```bash
git tag v0.1.0 && git push --tags
```

Then select the tag as the deployment branch in Streamlit Cloud settings.

---

## Local testing before deploy

```bash
cd platform
pip install -r requirements.txt
streamlit run app/main.py --server.port 8501
```

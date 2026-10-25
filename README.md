# COURTSIDE IQ — iPhone + Cloud deployment

This package contains the **same model.py and other simulation modules** as the COURTSIDE IQ Mac Quarter Stats build, with an adapted Streamlit `app.py` interface. It does not require the Mac to stay on once deployed.

## What is and is not done
- Ready for deployment to a Python/Streamlit hosting provider, including Streamlit Community Cloud and Docker-based hosts.
- Mobile CSS, browser-session preferences, and free-tier-friendly initial simulation settings are included.
- The full 150,000 simulation option remains available; free hosting might run out of memory, CPU or request time.
- NBA.com endpoints can rate-limit or block requests from cloud datacenter IPs. The app should display source errors rather than invent data.
- This ZIP **is not an active hosted website**. A hosting account must authorize deployment. No credentials or live domain are included.
- Cloud storage is ephemeral and is not synchronized with your Mac. Saved model training, archives, and caches can disappear on redeploy/restart. Multi-device sync is not implemented.

## Streamlit Community Cloud (free tier where available)
1. Sign in to GitHub and create a new repository; upload the files from this folder to the repository root (including `.streamlit/config.toml`).
2. Open https://share.streamlit.io/ and choose Create app from GitHub. Select your repository, branch, and `streamlit_app.py` as the entrypoint.
3. In advanced settings select Python 3.11 if available. The `streamlit_app.py` entrypoint sets cloud mode automatically; no secrets are needed.
4. Deploy, wait for dependencies, open the generated public URL in iPhone Safari, then Share → Add to Home Screen.

## Docker hosting
Build `docker build -t courtside-iq .` and run `docker run -p 8501:8501 courtside-iq`. A public URL, HTTPS, and hosting provider are still needed for access away from home.

## Reliability and privacy
- Keep the repository private if you do not want to publish the code. Verify the host supports deploying private repositories.
- Streamlit Community Cloud may sleep when unused and is not a persistent database.
- No automatic sync of saved history between Mac and cloud. Multiple visitors on one cloud instance share underlying disk caches; do not store personal secrets or sensitive data there.
- No accuracy guarantee: models are experimental, and live NBA.com endpoints can fail.

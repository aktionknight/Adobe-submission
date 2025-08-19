## Interface (Local)


IMP : if gemini 2.5 flash doesnt work, revert to 1.5 flash instead,
IMP : using azure TTS
IMP : Adobe api provided in doc attached along with this
(Docker server requests may be slow depending on hardware, local tests were returning 10-15 sec wait time for analysis)
(wait for around 30 sec - 45 sec for podcast generation, didnt get time to implement the loading bar for it)

FEATURES : 
CORE - 
   Multi PDF handling
   Select and analyse (press button after selection)
   Jump To location with a click
ADD ONS -
   Insight generation
   Multi voice Podcast Generation
BONUS FEATURES -
   Add a persona and a query to Get relevant sections (Optional)


Run a local web app that uses 1A-1B logic and Adobe PDF Embed.


Prerequisites:
- Python 3.9+ installed and on PATH
- Install dependencies used by 1B: `pip install -r ADOBE/1B/requirements.txt`
- Install interface dependencies: `pip install -r interface/requirements.txt`
- Get an Adobe PDF Embed API Client ID from the Adobe developer portal

/manual run down
Setup:
1. **Quick setup** (recommended): Run the setup script:
   ```bash
   python interface/setup_env.py
   ```
   This will prompt you for your Adobe API key and create the `.env` file automatically.

2. **Manual setup**: Create a `.env` file in the `interface/` directory:
   ```
   ADOBE_API=YOUR_ACTUAL_ADOBE_EMBED_CLIENT_ID_HERE
   ```

Start the server:

```bash
# Option 1: Use the startup script (recommended)
python interface/start_server.py

# Option 2: Run directly
python interface/backend/app.py
```

Open the app:
- Navigate to `http://127.0.0.1:8080/app`
- Upload PDFs, choose one to open, then click Analyze

Notes:
- Uploaded files are stored in `interface/uploads/`
- Top sections and related sections are computed with the same logic as `ADOBE/1B`
- Navigation to pages uses the Adobe PDF Embed API `gotoLocation`
- The backend is powered by FastAPI and serves both the API and static frontend files
- The Adobe API key is automatically loaded from the `.env` file and provided to the frontend 

GhostNote Python App

Runs the GhostNote UI in a browser while audio + MIDI are handled by a local Python engine.

Setup
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt

Run
  python app.py

Notes
  - The app opens http://127.0.0.1:8765/index.html and connects to a WebSocket on port 8766.
  - On macOS, it prefers opening in Google Chrome when installed.
  - If sounddevice fails to load on macOS, install PortAudio with: brew install portaudio
  - A click-to-open app bundle is at ghostnote_app/GhostNote.app (double-click it).
  - Logs live at ~/Library/Application Support/GhostNote/ghostnote.log

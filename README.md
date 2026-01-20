# GhostNote Bridge (Multichannel)

This package runs GhostNote in Chrome/Firefox while capturing multichannel audio using a local Python bridge.

## Quick Start (macOS)

1) Open Terminal and run:

```bash
cd /path/to/ghostnote_bridge
chmod +x run.sh
./run.sh
```

2) Open:

```
http://localhost:8000/index.html
```

## Requirements

- Python 3
- PortAudio (for multichannel input)

If audio fails to start, install PortAudio:

```bash
brew install portaudio
```

## Notes

- The bridge listens on `ws://127.0.0.1:8766`.
- Allow mic access for `python` in System Settings → Privacy & Security → Microphone.
- Use `PORT=8080 ./run.sh` to change the web server port.

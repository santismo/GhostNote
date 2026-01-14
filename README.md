# GhostNote 👻

GhostNote is a Chrome‑focused audio‑to‑MIDI drum brain for acoustic kits using piezo mics. It analyzes up to 16 input channels in real time, turns hits into MIDI notes, and routes them to any MIDI output (defaulting to Logic Pro Virtual In when available).

Live tool: https://santismo.github.io/GhostNote/

## Key Features

- **16‑channel input support** via multichannel audio interfaces
- **Per‑pad audio to MIDI** with thresholds, sensitivity, velocity curves, and masking
- **Pad layout editor** with draggable, non‑overlapping pads
- **Pad on/off toggles** to quickly mute a piece
- **Per‑pad filtering** (high‑pass / low‑pass) with graphical EQ overlay
- **Audio monitor** with waveform hang‑time and multiple views (Time, Rectified, Envelope, FFT)
- **Hi‑hat logic** (open/closed + choke) and **Smart Hi‑hat** mode
- **Per‑pad input boost (dB)** for balancing piezos
- **Global MIDI settings** (channel, note length, retrigger mask time)
- **Preset export/import** to JSON
- **Themes** with monochrome default + colored variants

## Requirements

- **Google Chrome** (Web Audio + Web MIDI)
- **Localhost or HTTPS** for audio permissions
- **Multichannel audio interface** if using multiple piezos

## Running Locally

Serve the file from localhost:

```bash
python3 -m http.server 8000
```

Then open:

```
http://localhost:8000
```

## Quick Start

1. **Turn Audio On** (top right button).
2. Choose your **Audio Input** device in the bottom bar.
3. Choose your **MIDI Output** (e.g., Logic Pro Virtual In).
4. Click any pad to open its **Pad Settings**.
5. Assign each pad to an **Input Channel** and **MIDI Note**.
6. Adjust **Threshold**, **Sensitivity**, **Mask Time**, and **Filters** as needed.

## Pad Settings

- **Input Channel**: which audio input feeds the pad
- **MIDI Note**: note number sent on trigger
- **Sensitivity**: scales velocity from peak amplitude
- **Input Boost (dB)**: pre‑gain for quieter piezos
- **Velocity Curve**: Linear, Soft, Hard
- **Threshold**: drag the right‑edge handle on the monitor
- **Mask Override**: per‑pad retrigger suppression
- **EQ**: drag HP/LP nodes over the monitor
- **Waveform View**: Time, Rectified, Envelope, FFT
- **Lock**: freeze all settings for that pad
- **On/Off**: toggle the pad directly on the kit

## Smart Hi‑hat

Smart Hi‑hat merges open/closed triggers into a single intent:

- **Enable** in the bottom bar.
- **Window (ms)**: if an open and closed hit occur within this window, only the closed note fires.
- Recommended setup with two piezos:
  - **Top cymbal piezo** → Open pad
  - **Bottom cymbal piezo** → Closed pad
  - Closed pad should have **Open Hat Choked By Closed = Yes**

## Global Controls (Bottom Bar)

- **Audio Input** selector
- **Channel Override** (if the OS reports fewer channels)
- **MIDI Output** selector
- **MIDI Channel** (1–16)
- **Note Length (ms)**
- **Mask Time (ms)**
- **Smart Hi‑hat** + **Window (ms)**
- **Theme** selector

## Presets

- **Save Preset** exports a JSON file with all settings and layout
- **Load Preset** imports a JSON file and restores the full state

## Tips

- Start with higher thresholds to avoid false triggers, then lower gradually.
- Use **Mask Time** to suppress double‑fires from piezo ringing.
- Use **Input Boost** to balance hotter/quieter piezos across the kit.
- For best latency, set a low buffer size in your audio interface control panel.


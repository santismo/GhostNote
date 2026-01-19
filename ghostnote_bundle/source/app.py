#!/usr/bin/env python3
import asyncio
import json
import threading
import time
import subprocess
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import mido
import numpy as np
import sounddevice as sd
import websockets

HTTP_PORT = 8765
WS_PORT = 8766
MAX_CHANNELS = 64
BLOCK_SIZE = 256
PUBLISH_INTERVAL = 0.03
THRESHOLD_MIN = 0.02
THRESHOLD_MAX = 1.5


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def db_to_gain(db):
    return 10 ** (db / 20)


class Biquad:
    def __init__(self, kind, freq, q, sample_rate):
        self.kind = kind
        self.sample_rate = float(sample_rate or 44100)
        self.b0 = 0.0
        self.b1 = 0.0
        self.b2 = 0.0
        self.a1 = 0.0
        self.a2 = 0.0
        self.z1 = 0.0
        self.z2 = 0.0
        self.update(freq, q)

    def update(self, freq, q):
        freq = clamp(float(freq), 10.0, self.sample_rate * 0.45)
        q = clamp(float(q), 0.1, 12.0)
        omega = 2 * np.pi * freq / self.sample_rate
        sin_w = np.sin(omega)
        cos_w = np.cos(omega)
        alpha = sin_w / (2 * q)

        if self.kind == "highpass":
            b0 = (1 + cos_w) / 2
            b1 = -(1 + cos_w)
            b2 = (1 + cos_w) / 2
        else:
            b0 = (1 - cos_w) / 2
            b1 = 1 - cos_w
            b2 = (1 - cos_w) / 2

        a0 = 1 + alpha
        a1 = -2 * cos_w
        a2 = 1 - alpha

        self.b0 = b0 / a0
        self.b1 = b1 / a0
        self.b2 = b2 / a0
        self.a1 = a1 / a0
        self.a2 = a2 / a0

    def process(self, samples):
        out = np.empty_like(samples)
        b0, b1, b2, a1, a2 = self.b0, self.b1, self.b2, self.a1, self.a2
        z1, z2 = self.z1, self.z2
        for i, x in enumerate(samples):
            y = b0 * x + z1
            z1 = b1 * x - a1 * y + z2
            z2 = b2 * x - a2 * y
            out[i] = y
        self.z1 = z1
        self.z2 = z2
        return out


@dataclass
class PadConfig:
    pad_id: str
    name: str
    input_channel: int
    midi_note: int
    threshold: float
    sensitivity: float
    velocity_curve: str
    mask_override: Optional[int]
    hihat_role: str
    hihat_choked_by_closed: bool
    hihat_choke_mode: str
    enabled: bool
    input_boost_db: float
    filter_hp: float
    filter_lp: float
    filter_hp_q: float
    filter_lp_q: float


@dataclass
class PadRuntime:
    highpass: Biquad
    lowpass: Biquad
    gain: float
    last_trigger_ms: float = 0.0
    meter: float = 0.0
    last_peak: float = 0.0
    note_off_timer: Optional[threading.Timer] = None
    active_note: Optional[int] = None
    learn_active: bool = False
    learn_max: float = 0.0
    learn_end_ms: float = 0.0


@dataclass
class SmartHihatState:
    pending_open_pad: Optional[str] = None
    pending_open_velocity: int = 0
    pending_open_due_ms: float = 0.0
    last_closed_ms: float = 0.0


class GhostNoteEngine:
    def __init__(self, loop):
        self.loop = loop
        self.clients = set()
        self.pads: Dict[str, PadConfig] = {}
        self.runtimes: Dict[str, PadRuntime] = {}
        self.selected_pad_id: Optional[str] = None
        self.smart_hihat = SmartHihatState()

        self.audio_active = False
        self.stream = None
        self.device_index = None
        self.sample_rate = 44100.0
        self.requested_channels = 0
        self.actual_channels = 0
        self.max_channels = 0
        self.detected_channels = 0

        self.midi_output_name = None
        self.midi_output = None
        self.midi_channel = 1
        self.note_length = 40
        self.mask_time = 80
        self.smart_hihat_enabled = False
        self.smart_hihat_window = 12
        self.channel_override = None

        self.state_lock = threading.Lock()
        self.wave_lock = threading.Lock()
        self.wave_samples = None
        self.wave_pad_id = None

        try:
            default_input = sd.default.device[0]
            if default_input is not None and default_input >= 0:
                self.device_index = int(default_input)
        except Exception:
            self.device_index = None

    def list_devices(self):
        devices = sd.query_devices()
        inputs = []
        for idx, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                inputs.append({
                    "id": idx,
                    "name": device["name"],
                    "maxChannels": device["max_input_channels"],
                })
        return inputs

    def list_midi_outputs(self):
        names = mido.get_output_names()
        return [{"id": name, "name": name} for name in names]

    def build_audio_status(self):
        return {
            "active": self.audio_active,
            "requested": self.requested_channels,
            "actual": self.actual_channels,
            "max": self.max_channels,
            "detected": self.detected_channels,
            "sampleRate": self.sample_rate,
        }

    async def broadcast(self, message):
        if not self.clients:
            return
        payload = json.dumps(message)
        dead = []
        for client in self.clients:
            try:
                await client.send(payload)
            except Exception:
                dead.append(client)
        for client in dead:
            self.clients.discard(client)

    def emit(self, message):
        if not self.clients:
            return
        asyncio.run_coroutine_threadsafe(self.broadcast(message), self.loop)

    def pad_from_payload(self, data):
        filt = data.get("filter") or {}
        mask_override = data.get("maskOverride")
        if mask_override is not None and mask_override != "":
            try:
                mask_override = int(mask_override)
            except (TypeError, ValueError):
                mask_override = None
        return PadConfig(
            pad_id=data["id"],
            name=data.get("name") or data["id"],
            input_channel=clamp(int(data.get("inputChannel", 1)), 1, MAX_CHANNELS),
            midi_note=clamp(int(data.get("midiNote", 36)), 0, 127),
            threshold=clamp(float(data.get("threshold", 0.12)), THRESHOLD_MIN, THRESHOLD_MAX),
            sensitivity=float(data.get("sensitivity", 1.0)),
            velocity_curve=data.get("velocityCurve", "linear"),
            mask_override=mask_override,
            hihat_role=data.get("hihatRole", "none"),
            hihat_choked_by_closed=bool(data.get("hihatChokedByClosed", False)),
            hihat_choke_mode=data.get("hihatChokeMode", "noteOff"),
            enabled=bool(data.get("enabled", True)),
            input_boost_db=float(data.get("inputBoostDb", 0)),
            filter_hp=float(filt.get("hp", 30)),
            filter_lp=float(filt.get("lp", 12000)),
            filter_hp_q=float(filt.get("hpQ", 0.7)),
            filter_lp_q=float(filt.get("lpQ", 0.7)),
        )

    def ensure_runtime(self, pad):
        runtime = self.runtimes.get(pad.pad_id)
        if runtime:
            runtime.highpass.sample_rate = self.sample_rate
            runtime.lowpass.sample_rate = self.sample_rate
            runtime.highpass.update(pad.filter_hp, pad.filter_hp_q)
            runtime.lowpass.update(pad.filter_lp, pad.filter_lp_q)
            runtime.gain = db_to_gain(pad.input_boost_db)
            return runtime
        runtime = PadRuntime(
            highpass=Biquad("highpass", pad.filter_hp, pad.filter_hp_q, self.sample_rate),
            lowpass=Biquad("lowpass", pad.filter_lp, pad.filter_lp_q, self.sample_rate),
            gain=db_to_gain(pad.input_boost_db),
        )
        self.runtimes[pad.pad_id] = runtime
        return runtime

    def update_global_settings(self, payload):
        if "midiChannel" in payload:
            self.midi_channel = clamp(int(payload.get("midiChannel", 1)), 1, 16)
        if "noteLength" in payload:
            self.note_length = clamp(int(payload.get("noteLength", 40)), 10, 2000)
        if "maskTime" in payload:
            self.mask_time = clamp(int(payload.get("maskTime", 80)), 10, 2000)
        if "channelOverride" in payload:
            override = payload.get("channelOverride")
            self.channel_override = int(override) if override not in (None, "", 0) else None
        if "smartHihatEnabled" in payload:
            self.smart_hihat_enabled = bool(payload.get("smartHihatEnabled"))
        if "smartHihatWindow" in payload:
            self.smart_hihat_window = clamp(int(payload.get("smartHihatWindow", 12)), 1, 200)

    def set_audio_device(self, device_id):
        if device_id in (None, ""):
            self.device_index = None
            return
        try:
            self.device_index = int(device_id)
            return
        except ValueError:
            pass
        for idx, device in enumerate(sd.query_devices()):
            if device_id == device["name"]:
                self.device_index = idx
                return

    def set_midi_output(self, output_id):
        if self.midi_output:
            try:
                self.midi_output.close()
            except Exception:
                pass
        self.midi_output = None
        self.midi_output_name = output_id
        if not output_id:
            return
        try:
            self.midi_output = mido.open_output(output_id)
        except Exception:
            self.midi_output = None

    def start_audio(self):
        self.stop_audio()
        device = None
        try:
            if self.device_index is not None:
                device = sd.query_devices(self.device_index, "input")
            else:
                default_input = sd.default.device[0]
                if default_input is not None and default_input >= 0:
                    device = sd.query_devices(default_input, "input")
        except Exception:
            device = None

        if device is None:
            self.emit({"type": "error", "payload": {"message": "No audio input device available."}})
            return

        max_channels = int(device["max_input_channels"] or 1)
        requested = self.channel_override or max_channels
        requested = clamp(requested, 1, MAX_CHANNELS)
        actual = clamp(min(requested, max_channels), 1, MAX_CHANNELS)
        self.requested_channels = requested
        self.actual_channels = actual
        self.max_channels = max_channels
        self.detected_channels = actual
        self.sample_rate = float(device.get("default_samplerate", 44100.0))
        with self.state_lock:
            for pad in self.pads.values():
                self.ensure_runtime(pad)

        try:
            self.stream = sd.InputStream(
                device=self.device_index,
                channels=actual,
                samplerate=self.sample_rate,
                blocksize=BLOCK_SIZE,
                dtype="float32",
                callback=self.audio_callback,
            )
            self.stream.start()
            self.audio_active = True
            self.emit({"type": "audio_status", "payload": self.build_audio_status()})
        except Exception as error:
            self.audio_active = False
            self.emit({"type": "error", "payload": {"message": f"Audio start failed: {error}"}})
            self.emit({"type": "audio_status", "payload": self.build_audio_status()})

    def stop_audio(self):
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.audio_active = False
        self.emit({"type": "audio_status", "payload": self.build_audio_status()})

    def restart_audio(self):
        self.stop_audio()
        self.start_audio()

    def audio_callback(self, indata, frames, time_info, status):
        now_ms = time.monotonic() * 1000
        with self.state_lock:
            pad_items = [(pad, self.runtimes.get(pad.pad_id)) for pad in self.pads.values()]
            selected_id = self.selected_pad_id

        wave_samples = None
        for pad, runtime in pad_items:
            if runtime is None:
                continue
            channel_index = pad.input_channel - 1
            if channel_index < 0 or channel_index >= indata.shape[1]:
                continue
            samples = indata[:, channel_index]
            filtered = runtime.highpass.process(samples)
            filtered = runtime.lowpass.process(filtered)
            if runtime.gain != 1.0:
                filtered = filtered * runtime.gain

            peak = float(np.max(np.abs(filtered))) if filtered.size else 0.0
            runtime.last_peak = peak
            runtime.meter = max(peak, runtime.meter * 0.92)

            if runtime.learn_active:
                runtime.learn_max = max(runtime.learn_max, peak)
                if now_ms >= runtime.learn_end_ms:
                    runtime.learn_active = False
                    pad.threshold = clamp(runtime.learn_max * 0.6, THRESHOLD_MIN, THRESHOLD_MAX)
                    self.emit({"type": "pad_update", "payload": {"pad": {"id": pad.pad_id, "threshold": pad.threshold}}})

            mask = pad.mask_override if pad.mask_override is not None else self.mask_time
            if pad.enabled and peak >= pad.threshold and now_ms - runtime.last_trigger_ms > mask:
                runtime.last_trigger_ms = now_ms
                self.handle_pad_trigger(pad, runtime, peak, now_ms)

            if pad.pad_id == selected_id:
                wave_samples = filtered.copy()

        if self.smart_hihat.pending_open_pad and now_ms >= self.smart_hihat.pending_open_due_ms:
            pad_id = self.smart_hihat.pending_open_pad
            velocity = self.smart_hihat.pending_open_velocity
            self.smart_hihat.pending_open_pad = None
            pad = self.pads.get(pad_id)
            runtime = self.runtimes.get(pad_id)
            if pad and runtime and pad.enabled:
                self.send_note(pad, runtime, velocity)
                self.emit({"type": "pad_hit", "payload": {"id": pad_id, "velocity": velocity, "peak": runtime.last_peak}})

        if wave_samples is not None:
            with self.wave_lock:
                self.wave_samples = wave_samples
                self.wave_pad_id = selected_id

    def handle_pad_trigger(self, pad, runtime, peak, now_ms):
        velocity = self.calculate_velocity(peak, pad.sensitivity, pad.velocity_curve)
        if self.smart_hihat_enabled and pad.hihat_role in ("open", "closed"):
            if pad.hihat_role == "open":
                if now_ms - self.smart_hihat.last_closed_ms <= self.smart_hihat_window:
                    return
                self.smart_hihat.pending_open_pad = pad.pad_id
                self.smart_hihat.pending_open_velocity = velocity
                self.smart_hihat.pending_open_due_ms = now_ms + self.smart_hihat_window
                return
            self.smart_hihat.last_closed_ms = now_ms
            if self.smart_hihat.pending_open_pad and now_ms <= self.smart_hihat.pending_open_due_ms:
                self.smart_hihat.pending_open_pad = None
            self.choke_open_hats(pad)

        self.send_note(pad, runtime, velocity)
        self.emit({"type": "pad_hit", "payload": {"id": pad.pad_id, "velocity": velocity, "peak": peak}})

    def choke_open_hats(self, trigger_pad):
        with self.state_lock:
            pads = list(self.pads.values())
        for pad in pads:
            if pad.hihat_role != "open" or not pad.hihat_choked_by_closed:
                continue
            runtime = self.runtimes.get(pad.pad_id)
            if not runtime or runtime.active_note is None:
                continue
            if pad.hihat_choke_mode == "noteOnZero" or trigger_pad.hihat_choke_mode == "noteOnZero":
                self.send_note_on(runtime.active_note, 0)
            else:
                self.send_note_off(runtime.active_note)
            if runtime.note_off_timer:
                runtime.note_off_timer.cancel()
            runtime.active_note = None

    def send_note_on(self, note, velocity):
        if not self.midi_output:
            return
        message = mido.Message("note_on", note=note, velocity=velocity, channel=self.midi_channel - 1)
        self.midi_output.send(message)

    def send_note_off(self, note):
        if not self.midi_output:
            return
        message = mido.Message("note_off", note=note, velocity=0, channel=self.midi_channel - 1)
        self.midi_output.send(message)

    def send_note(self, pad, runtime, velocity):
        if runtime.note_off_timer:
            runtime.note_off_timer.cancel()
        self.send_note_on(pad.midi_note, velocity)
        runtime.active_note = pad.midi_note
        runtime.note_off_timer = threading.Timer(self.note_length / 1000.0, self.send_note_off, args=(pad.midi_note,))
        runtime.note_off_timer.start()

    def calculate_velocity(self, peak, sensitivity, curve):
        value = clamp(peak * sensitivity, 0.0, 1.0)
        if curve == "soft":
            value = np.sqrt(value)
        elif curve == "hard":
            value = value * value
        velocity = int(round(value * 127))
        return clamp(velocity, 1, 127)

    async def publish_loop(self):
        while True:
            await asyncio.sleep(PUBLISH_INTERVAL)
            if not self.audio_active:
                continue
            meters = []
            with self.state_lock:
                pad_items = list(self.pads.values())
            with self.wave_lock:
                wave_samples = self.wave_samples
                wave_pad_id = self.wave_pad_id

            for pad in pad_items:
                runtime = self.runtimes.get(pad.pad_id)
                if not runtime:
                    continue
                meters.append({
                    "id": pad.pad_id,
                    "peak": runtime.last_peak,
                    "meter": runtime.meter,
                })
            if meters:
                await self.broadcast({"type": "pad_meter", "payload": {"meters": meters}})

            if wave_samples is not None and wave_pad_id:
                samples = np.asarray(wave_samples, dtype=np.float32)
                if samples.size > 256:
                    idx = np.linspace(0, samples.size - 1, 256)
                    samples = np.interp(idx, np.arange(samples.size), samples)
                fft = np.abs(np.fft.rfft(samples))
                if fft.size:
                    peak = np.max(fft) or 1.0
                    fft = fft / peak
                await self.broadcast({
                    "type": "waveform",
                    "payload": {
                        "id": wave_pad_id,
                        "samples": samples.tolist(),
                        "fft": fft.tolist(),
                    },
                })

    async def handle_message(self, websocket, message):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return
        msg_type = data.get("type")
        payload = data.get("payload") or {}

        if msg_type == "hello":
            await websocket.send(json.dumps({
                "type": "init",
                "payload": {
                    "devices": self.list_devices(),
                    "midiOutputs": self.list_midi_outputs(),
                    "audioStatus": self.build_audio_status(),
                    "selectedDeviceId": self.device_index,
                    "selectedMidiOutputId": self.midi_output_name,
                },
            }))
            return

        if msg_type == "list_devices":
            await websocket.send(json.dumps({"type": "devices", "payload": {"devices": self.list_devices()}}))
            return

        if msg_type == "list_midi":
            await websocket.send(json.dumps({"type": "midi", "payload": {"outputs": self.list_midi_outputs()}}))
            return

        if msg_type == "set_audio_device":
            self.set_audio_device(payload.get("id"))
            if self.audio_active:
                self.restart_audio()
            return

        if msg_type == "set_midi_output":
            self.set_midi_output(payload.get("id"))
            await websocket.send(json.dumps({"type": "midi", "payload": {"outputs": self.list_midi_outputs()}}))
            return

        if msg_type == "toggle_audio":
            if self.audio_active:
                self.stop_audio()
            else:
                self.start_audio()
            return

        if msg_type == "start_audio":
            self.start_audio()
            return

        if msg_type == "stop_audio":
            self.stop_audio()
            return

        if msg_type == "restart_audio":
            self.restart_audio()
            return

        if msg_type == "select_pad":
            with self.state_lock:
                self.selected_pad_id = payload.get("id") or None
            return

        if msg_type == "pad_learn":
            pad_id = payload.get("id")
            with self.state_lock:
                runtime = self.runtimes.get(pad_id)
                if runtime:
                    runtime.learn_active = True
                    runtime.learn_max = 0.0
                    runtime.learn_end_ms = time.monotonic() * 1000 + 1500
            return

        if msg_type == "pad_remove":
            pad_id = payload.get("id")
            if pad_id:
                with self.state_lock:
                    self.pads.pop(pad_id, None)
                    self.runtimes.pop(pad_id, None)
            return

        if msg_type == "pad_update":
            pad_data = payload.get("pad")
            if pad_data:
                pad = self.pad_from_payload(pad_data)
                with self.state_lock:
                    self.pads[pad.pad_id] = pad
                    self.ensure_runtime(pad)
            return

        if msg_type == "global_update":
            self.update_global_settings(payload)
            if self.audio_active and "channelOverride" in payload:
                self.restart_audio()
            return

        if msg_type == "sync_state":
            pads = payload.get("pads") or []
            with self.state_lock:
                incoming_ids = set()
                for pad_data in pads:
                    pad = self.pad_from_payload(pad_data)
                    incoming_ids.add(pad.pad_id)
                    self.pads[pad.pad_id] = pad
                    self.ensure_runtime(pad)
                for pad_id in list(self.pads.keys()):
                    if pad_id not in incoming_ids:
                        self.pads.pop(pad_id, None)
                        self.runtimes.pop(pad_id, None)
                self.selected_pad_id = payload.get("selectedPadId") or None
            self.update_global_settings(payload.get("global") or {})
            if "audioDeviceId" in payload:
                self.set_audio_device(payload.get("audioDeviceId"))
            if "midiOutputId" in payload:
                self.set_midi_output(payload.get("midiOutputId"))
            return

    async def handle_client(self, websocket):
        self.clients.add(websocket)
        await websocket.send(json.dumps({
            "type": "init",
            "payload": {
                "devices": self.list_devices(),
                "midiOutputs": self.list_midi_outputs(),
                "audioStatus": self.build_audio_status(),
                "selectedDeviceId": self.device_index,
                "selectedMidiOutputId": self.midi_output_name,
            },
        }))
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        finally:
            self.clients.discard(websocket)


def start_http_server(root_dir, host="127.0.0.1", port=HTTP_PORT):
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    class SilentHandler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            return

    handler = lambda *args, **kwargs: SilentHandler(*args, directory=str(root_dir), **kwargs)
    httpd = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def open_ui(url):
    if sys.platform == "darwin":
        chrome_path = "/Applications/Google Chrome.app"
        if Path(chrome_path).exists():
            try:
                subprocess.run(["open", "-a", "Google Chrome", url], check=False)
                return
            except Exception:
                pass
    webbrowser.open(url)


async def main():
    loop = asyncio.get_running_loop()
    engine = GhostNoteEngine(loop)
    static_dir = Path(__file__).parent / "static"
    start_http_server(static_dir)
    open_ui(f"http://127.0.0.1:{HTTP_PORT}/index.html")

    async with websockets.serve(engine.handle_client, "127.0.0.1", WS_PORT):
        await engine.publish_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

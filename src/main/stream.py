#!/usr/bin/env python3

import pyaudio
from pylsl import StreamInfo, StreamOutlet
import numpy as np
import socket

# Constants
FORMAT = pyaudio.paInt24  # 24-bit integer samples
CHANNELS = 2  # Stereo input
RATE = 48000  # Sample rate in Hz
CHUNK = int(RATE * 0.1)  # Buffer size (100ms of audio)

# Initialize PyAudio
p = pyaudio.PyAudio()

# Find the Yeti X microphone on Host API 2 or 3
device_index = None
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if (
        "Yeti X" in dev["name"]
        and dev["hostApi"] in [2, 3]
        and dev["maxInputChannels"] > 0
    ):
        # Verify if the device supports 24-bit format at the desired sample rate
        if p.is_format_supported(
            RATE,
            input_device=dev["index"],
            input_channels=CHANNELS,
            input_format=FORMAT,
        ):
            device_index = i
            break

if device_index is None:
    print("Yeti X microphone supporting 24-bit audio not found on Host API 2 or 3.")
    p.terminate()
    exit(1)

print(f"Recording from: {p.get_device_info_by_index(device_index)['name']}")

# Open the audio stream
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
    input_device_index=device_index,
)

# Create LSL stream info and outlet
info = StreamInfo(
    "YetiX_Audio",
    "Audio",
    CHANNELS,
    RATE,
    "float32",
    "audio_stream_" + socket.gethostname(),
)
outlet = StreamOutlet(info)

print("RECORDING ---------- Press Ctrl+C to stop.")

try:
    while True:
        # Read raw data from the microphone
        data = stream.read(CHUNK, exception_on_overflow=False)

        # Convert 24-bit data to float32
        nump = np.frombuffer(data, dtype=np.uint8)
        nump = nump.reshape(-1, 3)

        # Convert bytes to int32
        int_samples = (
            nump[:, 0].astype(np.int32)
            | (nump[:, 1].astype(np.int32) << 8)
            | (nump[:, 2].astype(np.int32) << 16)
        )

        # Sign extension for negative values
        sign_mask = 1 << 23
        int_samples = (int_samples ^ sign_mask) - sign_mask

        # Normalize to [-1.0, 1.0] float32
        float_samples = int_samples / 8388608.0  # 2^23

        # Reshape data to match the number of channels
        float_samples = float_samples.reshape(-1, CHANNELS)

        # Send data via LSL
        outlet.push_chunk(float_samples.tolist())
except KeyboardInterrupt:
    print("\nSTOPPING...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Audio stream closed.")

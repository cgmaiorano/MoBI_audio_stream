# src/audio_recorder/recorder.py

import pyaudio
from pylsl import StreamInfo, StreamOutlet
import numpy as np
import socket
import json
import os
import time
import logging

# Configure logging
LOG_FILE = os.path.join(os.path.dirname(__file__), "audio_recorder.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
)

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

if not os.path.exists(CONFIG_PATH):
    print(f"Configuration file {CONFIG_PATH} not found.")
    exit(1)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Constants from config
DEVICE_NAME = config.get("device_name", "Yeti X")
HOST_API = config.get("host_api", [2, 3])
CHANNELS = config.get("channels", 2)
RATE = config.get("sample_rate", 48000)
CHUNK = int(RATE * config.get("chunk_duration", 0.1))
FORMAT = getattr(pyaudio, config.get("format", "paInt24"))
LSL_STREAM_NAME = config.get("lsl_stream_name", "YetiX_Audio")
LSL_STREAM_TYPE = config.get("lsl_stream_type", "Audio")

# Control flag file
FLAG_FILE = os.path.join(os.path.dirname(__file__), "recording.flag")


def find_device(p):
    """Find the audio device index based on the device name and host API."""
    device_index = None
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if (
            DEVICE_NAME in dev["name"]
            and dev["hostApi"] in HOST_API
            and dev["maxInputChannels"] > 0
        ):
            if p.is_format_supported(
                RATE,
                input_device=dev["index"],
                input_channels=CHANNELS,
                input_format=FORMAT,
            ):
                device_index = i
                break
    return device_index


def start_recording(p, device_index):
    """Start the audio stream and create the LSL outlet."""
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=device_index,
        )
    except Exception as e:
        logging.error("Failed to open audio stream.", exc_info=True)
        print("Failed to open audio stream. Please check the device and try again.")
        return None, None

    # Create LSL stream info and outlet
    info = StreamInfo(
        LSL_STREAM_NAME,
        LSL_STREAM_TYPE,
        CHANNELS,
        RATE,
        "float32",
        "audio_stream_" + socket.gethostname(),
    )
    outlet = StreamOutlet(info)

    logging.info("LSL outlet created. Starting the recording loop.")

    return stream, outlet


def stop_recording(stream, p):
    """Stop the audio stream and terminate PyAudio."""
    stream.stop_stream()
    stream.close()
    p.terminate()
    logging.info("Audio stream closed.")


def record_audio():
    """Record audio and stream it via LSL while the flag file exists."""
    p = pyaudio.PyAudio()
    device_index = find_device(p)

    if device_index is None:
        available_devices = [
            p.get_device_info_by_index(i)["name"] for i in range(p.get_device_count())
        ]
        message = (
            f"Device '{DEVICE_NAME}' not found.\nAvailable devices:\n"
            + "\n".join(available_devices)
        )
        logging.error(message)
        print(message)
        p.terminate()
        return

    logging.info(f"Recording from: {p.get_device_info_by_index(device_index)['name']}")
    stream, outlet = start_recording(p, device_index)
    if stream is None:
        return

    try:
        while os.path.exists(FLAG_FILE):
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
    except Exception as e:
        logging.error("An error occurred during recording.", exc_info=True)
        print("An error occurred during recording. Please check the log file.")
    finally:
        stop_recording(stream, p)
        logging.info("Recording stopped.")


def main():
    """Main function that monitors the flag file to start and stop recording."""
    print("Audio Recorder is running. Waiting for the recording flag...")
    logging.info("Audio Recorder started.")

    try:
        while True:
            if os.path.exists(FLAG_FILE):
                logging.info("Recording flag detected. Starting recording.")
                print("Recording started.")
                record_audio()
                logging.info("Recording session ended.")
                print("Recording session ended. Waiting for the next recording flag...")
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Audio Recorder terminated by user.")
        print("\nAudio Recorder terminated.")
    except Exception as e:
        logging.error("An unexpected error occurred.", exc_info=True)
        print("An unexpected error occurred. Please check the log file.")


if __name__ == "__main__":
    main()

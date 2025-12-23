#!/usr/bin/env python3

import os
import sys
import json
import shutil
import threading
import signal
import subprocess
import time
import argparse
import re
import glob

import yaml
from pynput import keyboard
from pynput.keyboard import Key, Controller

# Try to import both engines - will be used based on runtime selection
try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    import numpy as np
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "config.yaml")

# Load config early to get default engine
with open(CONFIG_FILE) as f:
    _temp_config = yaml.safe_load(f)
    ENGINE = _temp_config.get("engine", "vosk")

SAMPLE_RATE = 16000

# Key name mapping
KEY_MAP = {
    "alt_l": Key.alt_l,
    "alt_r": Key.alt_r,
    "ctrl_l": Key.ctrl_l,
    "ctrl_r": Key.ctrl_r,
    "shift_l": Key.shift_l,
    "shift_r": Key.shift_r,
    "scroll_lock": Key.scroll_lock,
    "pause": Key.pause,
    "insert": Key.insert,
    "delete": Key.delete,
}

# Known Whisper models
WHISPER_MODELS = [
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large", "large-v1", "large-v2", "large-v3"
]


def load_config():
    """Load configuration from YAML file"""
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


config = load_config()
trigger_key_name = config.get("key", "alt_r")

# Parse trigger key - support combinations like "shift-ctrl" or single keys like "alt_r"
if "-" in trigger_key_name:
    # Key combination
    key_parts = [k.strip() for k in trigger_key_name.split("-")]
    TRIGGER_KEYS = set(KEY_MAP.get(k, KEY_MAP.get(k.lower())) for k in key_parts)
else:
    # Single key
    TRIGGER_KEYS = {KEY_MAP.get(trigger_key_name, Key.alt_r)}

MODEL_NAME = config.get("model", "vosk-model-small-en-us-0.15")
# Normalize voice command keys to lowercase for case-insensitive matching
VOICE_COMMANDS = {k.lower(): v for k, v in config.get("commands", {}).items()}
TYPING_MODE = config.get("mode", "buffered")  # buffered or realtime
PAUSE_DELAY = config.get("pause", 0.3)

# Global state
is_recording = False
has_typed_anything = False
capitalize_next = True  # Capitalize first word and after sentence-ending punctuation
last_char_typed = ""  # Track last character to prevent double spaces
currently_pressed_keys = set()  # Track pressed keys for combination support
lock = threading.Lock()
kb_controller = Controller()
model = None  # Vosk model
whisper_model = None  # Whisper model
recording_thread = None
stop_recording_event = threading.Event()


def check_dependencies():
    """Check if required dependencies are installed"""
    missing = []

    if shutil.which("arecord") is None:
        missing.append("arecord (install with: sudo apt install alsa-utils)")

    if missing:
        print("Missing dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        return False

    return True


def get_available_vosk_models():
    """Get list of downloaded Vosk models in the script directory"""
    model_dirs = glob.glob(os.path.join(SCRIPT_DIR, "vosk-model-*"))
    return [os.path.basename(d) for d in model_dirs if os.path.isdir(d)]


def resolve_model_name(pattern, engine):
    """
    Resolve a model name pattern to a full model name.
    Pattern can be:
    - Exact match: returns as-is if it matches exactly
    - Regex pattern: matches against available models

    For Vosk: matches against downloaded models in the directory
    For Whisper: matches against known Whisper model names
    """
    if engine == "vosk":
        available = get_available_vosk_models()
    elif engine == "whisper":
        available = WHISPER_MODELS
    else:
        return pattern

    # First try exact match
    if pattern in available:
        return pattern

    # Try regex match
    try:
        regex = re.compile(pattern, re.IGNORECASE)
        matches = [m for m in available if regex.search(m)]

        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            print(f"Pattern '{pattern}' matches multiple models:")
            for m in matches:
                print(f"  - {m}")
            print(f"Using first match: {matches[0]}")
            return matches[0]
        else:
            print(f"Pattern '{pattern}' does not match any available models.")
            if engine == "vosk":
                print(f"Available Vosk models: {', '.join(available) if available else 'none (download first)'}")
            else:
                print(f"Available Whisper models: {', '.join(available)}")
            return pattern
    except re.error as e:
        print(f"Invalid regex pattern '{pattern}': {e}")
        return pattern


def process_voice_commands(words):
    """Convert voice command words to punctuation and symbols"""
    result = []
    i = 0
    while i < len(words):
        matched = False
        # Try matching longest phrases first (2 words, then 1 word)
        for length in [2, 1]:
            if i + length <= len(words):
                phrase = " ".join(words[i:i+length]).lower()
                if phrase in VOICE_COMMANDS:
                    command_output = VOICE_COMMANDS[phrase]
                    next_idx = i + length

                    # If command produces punctuation and next token is any punctuation
                    if command_output in ".,?!:;" and next_idx < len(words) and words[next_idx] in ".,?!:;":
                        # Whisper added punctuation (might be different), skip it and use command output
                        if not (result and result[-1] == command_output):
                            result.append(command_output)
                        i = next_idx + 1  # Skip both command word and Whisper's punctuation
                    elif next_idx < len(words) and words[next_idx] == command_output:
                        # Whisper added the exact same punctuation, skip command word
                        # Next iteration will pick up the punctuation from Whisper
                        i += length
                    else:
                        # No following punctuation, add command output
                        if not (command_output in ".,?!:;" and result and result[-1] == command_output):
                            result.append(command_output)
                        i += length
                    matched = True
                    break
        if not matched:
            # Skip if this is duplicate consecutive punctuation
            if words[i] in ".,?!:;" and result and result[-1] == words[i]:
                i += 1
            else:
                result.append(words[i])
                i += 1
    return result


def type_text(words):
    """Type words at current cursor position"""
    global has_typed_anything, capitalize_next, last_char_typed

    if not words:
        return

    processed = process_voice_commands(words)

    for word in processed:
        is_punctuation = word in ".,?!:;"
        is_sentence_end = word in ".?!"

        if is_punctuation:
            # Punctuation: no space before, space after
            kb_controller.type(word + " ")
            last_char_typed = " "
            has_typed_anything = False  # Next word shouldn't have leading space
            if is_sentence_end:
                capitalize_next = True
        else:
            # Capitalize "I" pronoun
            if word.lower() == "i":
                word = "I"
            # Capitalize first letter if needed
            elif capitalize_next and word:
                word = word[0].upper() + word[1:]

            if capitalize_next:
                capitalize_next = False

            if has_typed_anything:
                # Regular word: space before
                kb_controller.type(" " + word)
                last_char_typed = word[-1] if word else ""
            else:
                # First word (or after punctuation): no space before
                kb_controller.type(word)
                last_char_typed = word[-1] if word else ""
                has_typed_anything = True


def stream_transcribe():
    """Record and transcribe audio, typing based on configured mode"""
    global model

    rec = KaldiRecognizer(model, SAMPLE_RATE)

    # Start arecord process
    process = subprocess.Popen([
        "arecord",
        "-f", "S16_LE",
        "-r", str(SAMPLE_RATE),
        "-c", "1",
        "-t", "raw",
        "-q"
    ], stdout=subprocess.PIPE)

    last_partial_words = []

    try:
        while not stop_recording_event.is_set():
            data = process.stdout.read(4000)
            if len(data) == 0:
                break

            if rec.AcceptWaveform(data):
                # Final result - Vosk has detected a phrase boundary (pause)
                result = json.loads(rec.Result())
                text = result.get("text", "")
                if text:
                    if TYPING_MODE == "buffered":
                        # Buffered mode: type the complete final result with optional delay
                        if PAUSE_DELAY > 0:
                            time.sleep(PAUSE_DELAY)
                        final_words = text.split()
                        type_text(final_words)
                    else:
                        # Realtime mode: type any new words not already typed
                        final_words = text.split()
                        new_words = final_words[len(last_partial_words):]
                        if new_words:
                            type_text(new_words)
                    last_partial_words = []
            else:
                # Partial result - intermediate prediction
                if TYPING_MODE == "realtime":
                    # Realtime mode: type new words as they appear
                    partial = json.loads(rec.PartialResult())
                    partial_text = partial.get("partial", "")
                    if partial_text:
                        partial_words = partial_text.split()
                        new_words = partial_words[len(last_partial_words):]
                        if new_words:
                            type_text(new_words)
                            last_partial_words = partial_words
                # Buffered mode: ignore partials, wait for final results

        # Get any remaining audio as final result
        result = json.loads(rec.FinalResult())
        text = result.get("text", "")
        if text:
            final_words = text.split()
            if TYPING_MODE == "buffered":
                # Type the complete final result
                type_text(final_words)
            else:
                # Realtime mode: only type new words
                new_words = final_words[len(last_partial_words):]
                if new_words:
                    type_text(new_words)

    finally:
        process.terminate()
        process.wait()


def stream_transcribe_whisper():
    """Record and transcribe audio using faster-whisper with VAD"""
    global whisper_model

    # Start arecord process (same as Vosk)
    process = subprocess.Popen([
        "arecord",
        "-f", "S16_LE",
        "-r", str(SAMPLE_RATE),
        "-c", "1",
        "-t", "raw",
        "-q"
    ], stdout=subprocess.PIPE)

    audio_chunks = []

    try:
        while not stop_recording_event.is_set():
            data = process.stdout.read(4000)
            if len(data) == 0:
                break

            # Convert S16_LE to float32 numpy array
            audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_chunks.append(audio_chunk)

        # Transcribe accumulated audio when key is released
        if audio_chunks:
            audio_data = np.concatenate(audio_chunks)

            # Skip if recording is too short (less than 0.5 seconds)
            if len(audio_data) >= SAMPLE_RATE * 0.5:
                segments, _ = whisper_model.transcribe(
                    audio_data,
                    language="en",
                    vad_filter=True
                )

                # Collect all segment text
                text = " ".join(segment.text.strip() for segment in segments)

                if text:
                    # Clean up Whisper quirks
                    text = text.replace(",,", ",")  # Multiple commas to single comma
                    if text.endswith("..."):
                        text = text[:-3]  # Remove trailing ellipsis

                    # Split words and separate punctuation for voice command matching
                    tokens = []
                    for word in text.split():
                        # Separate trailing punctuation from word
                        stripped = word.rstrip(".,?!:;")
                        trailing = word[len(stripped):]
                        if stripped:
                            tokens.append(stripped)
                        if trailing:
                            # Add each punctuation character separately
                            tokens.extend(list(trailing))

                    if tokens:
                        type_text(tokens)
                        # Add space after transcribed text only if it doesn't end with punctuation
                        # and we didn't just type a space (prevent double spaces)
                        last_token = tokens[-1]
                        if last_token not in (".", ",", "?", "!", ":", ";") and last_char_typed != " ":
                            kb_controller.type(" ")
                            last_char_typed = " "

    finally:
        process.terminate()
        process.wait()


def on_key_press(key):
    """Handle key press events"""
    global is_recording, recording_thread, has_typed_anything, capitalize_next, last_char_typed, currently_pressed_keys

    # Track pressed keys for combination detection
    currently_pressed_keys.add(key)

    # Check if all trigger keys are now pressed
    if TRIGGER_KEYS.issubset(currently_pressed_keys):
        with lock:
            if not is_recording:
                is_recording = True
                has_typed_anything = False
                capitalize_next = True  # Start new recording with capitalization
                last_char_typed = ""  # Reset for new recording
                stop_recording_event.clear()

                # Select transcription function based on engine
                transcribe_fn = stream_transcribe_whisper if ENGINE == "whisper" else stream_transcribe
                recording_thread = threading.Thread(target=transcribe_fn, daemon=True)
                recording_thread.start()


def on_key_release(key):
    """Handle key release events"""
    global is_recording, recording_thread, currently_pressed_keys

    # Remove from pressed keys
    currently_pressed_keys.discard(key)

    # If we were recording and any trigger key was released, stop recording
    if is_recording and key in TRIGGER_KEYS:
        with lock:
            if is_recording:
                is_recording = False
                stop_recording_event.set()
                if recording_thread:
                    recording_thread.join(timeout=1.0)
                    recording_thread = None


def main():
    global model, whisper_model, TRIGGER_KEYS, TYPING_MODE, PAUSE_DELAY, ENGINE

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Voice-to-keyboard: Hold a key to record, text appears as you speak",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --engine whisper --key alt_r
  %(prog)s --engine vosk --key shift_l-ctrl_r
  %(prog)s --engine whisper --key ctrl_l --mode realtime
  %(prog)s --engine whisper --key alt_r --model medium.en

Available keys: """ + ", ".join(sorted(set(KEY_MAP.keys()))) + """

Key combinations: Use '-' to combine keys (e.g., shift_l-ctrl_l, shift_l-alt_r)
"""
    )

    parser.add_argument("--engine", choices=["vosk", "whisper"], required=True,
                        help="Speech recognition engine (required)")
    parser.add_argument("--model",
                        help="Model name (default: from config)")
    parser.add_argument("--key", required=True,
                        help="Trigger key or combination (e.g., 'alt_r', 'shift_l-ctrl_r') (required)")
    parser.add_argument("--mode", dest="typing_mode", choices=["buffered", "realtime"],
                        help="Typing mode (default: from config)")
    parser.add_argument("--pause", type=float, metavar="SECONDS",
                        help="Pause delay in seconds for buffered mode (default: from config)")

    args = parser.parse_args()

    # Engine is required from command line
    engine = args.engine
    ENGINE = engine  # Update global so on_key_press uses the right engine

    # Resolve model name (support regex matching)
    if args.model:
        model_name = resolve_model_name(args.model, engine)
    else:
        # Use engine-specific default if config has wrong engine's model
        if engine == "whisper" and (not MODEL_NAME or MODEL_NAME.startswith("vosk-")):
            model_name = "small.en"  # Default Whisper model
        elif engine == "vosk" and MODEL_NAME and not MODEL_NAME.startswith("vosk-"):
            model_name = "vosk-model-small-en-us-0.15"  # Default Vosk model
        else:
            model_name = MODEL_NAME

    # Parse trigger key - support combinations like "shift-ctrl_r"
    if "-" in args.key:
        key_parts = [k.strip() for k in args.key.split("-")]
        TRIGGER_KEYS = set(KEY_MAP.get(k, KEY_MAP.get(k.lower())) for k in key_parts)
    else:
        TRIGGER_KEYS = {KEY_MAP.get(args.key, KEY_MAP.get(args.key.lower()))}
    if args.typing_mode:
        TYPING_MODE = args.typing_mode
    if args.pause is not None:
        PAUSE_DELAY = args.pause

    if not check_dependencies():
        return 1

    # Check if requested engine is available
    if engine == "vosk" and not VOSK_AVAILABLE:
        print(f"Error: Vosk engine selected but not installed")
        print(f"Run: pip install vosk")
        return 1
    elif engine == "whisper" and not WHISPER_AVAILABLE:
        print(f"Error: Whisper engine selected but not installed")
        print(f"Run: pip install faster-whisper numpy")
        return 1

    if engine == "vosk":
        model_path = os.path.join(SCRIPT_DIR, model_name)

        if not os.path.exists(model_path):
            print(f"Vosk model not found at {model_path}")
            return 1

        print(f"Loading Vosk model ({model_name})...")
        model = Model(model_path)

    elif engine == "whisper":
        print(f"Loading Whisper model ({model_name})...")
        whisper_model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8"
        )
        print(f"Whisper model loaded")

        # Warn if realtime mode is set with Whisper
        if TYPING_MODE == "realtime":
            print("Warning: realtime mode not supported with Whisper engine, using buffered mode")

    else:
        print(f"Unknown engine: {engine}")
        return 1

    # Format trigger keys display
    if len(TRIGGER_KEYS) > 1:
        trigger_display = "-".join(str(k) for k in TRIGGER_KEYS)
    else:
        trigger_display = str(list(TRIGGER_KEYS)[0])

    print("voice2keyboard running")
    print(f"Engine: {engine}")
    print(f"Hold {trigger_display} to record")
    print(f"Mode: {TYPING_MODE}" + (f" (pause_delay: {PAUSE_DELAY}s)" if TYPING_MODE == "buffered" and PAUSE_DELAY > 0 else ""))
    print("Press Ctrl+C to exit")

    def signal_handler(sig, frame):
        print("\nExiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    with keyboard.Listener(on_press=on_key_press, on_release=on_key_release) as listener:
        listener.join()

    return 0


if __name__ == "__main__":
    sys.exit(main())

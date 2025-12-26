#!/usr/bin/env python3

import os
import sys
import json
import shutil
import threading
import signal
import subprocess
import time
from datetime import datetime
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

SAMPLE_RATE = 16000
ENGINE = None  # Set via command-line argument

# Key name mapping
KEY_MAP = {
    "alt_l": Key.alt_l,
    "alt_r": Key.alt_r,
    "ctrl_l": Key.ctrl_l,
    "ctrl_r": Key.ctrl_r,
    "ctl_l": Key.ctrl_l,  # Alias for ctrl_l
    "ctl_r": Key.ctrl_r,  # Alias for ctrl_r
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
    "large", "large-v1", "large-v2", "large-v3",
    # Distil-Whisper models (5-6x faster, similar accuracy)
    "distil-small.en",
    "distil-medium.en",
    "distil-large-v2",
    "distil-large-v3",
]


def load_config():
    """Load configuration from YAML file"""
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


config = load_config()
# Normalize voice translation keys to lowercase for case-insensitive matching
VOICE_TRANSLATIONS = {k.lower(): v for k, v in config.get("vosk-translations", {}).items()}
TYPING_MODE = config.get("mode", "buffered")  # buffered or realtime
PAUSE_DELAY = config.get("pause", 0.3)

# Global state
ENGINE = None  # Current speech engine (vosk or whisper)
key_release_time = None  # Track when user released the key
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


def process_voice_translations(words):
    """Convert voice translation words to punctuation and symbols"""
    result = []
    i = 0
    while i < len(words):
        matched = False
        # Try matching longest phrases first (2 words, then 1 word)
        for length in [2, 1]:
            if i + length <= len(words):
                phrase = " ".join(words[i:i+length]).lower()
                if phrase in VOICE_TRANSLATIONS:
                    translation_output = VOICE_TRANSLATIONS[phrase]
                    next_idx = i + length

                    # If translation produces punctuation and next token is any punctuation
                    if translation_output in ".,?!:;" and next_idx < len(words) and words[next_idx] in ".,?!:;":
                        # Whisper added punctuation (might be different), skip it and use translation output
                        if not (result and result[-1] == translation_output):
                            result.append(translation_output)
                        i = next_idx + 1  # Skip both translation word and Whisper's punctuation
                    elif next_idx < len(words) and words[next_idx] == translation_output:
                        # Whisper added the exact same punctuation, skip translation word
                        # Next iteration will pick up the punctuation from Whisper
                        i += length
                    else:
                        # No following punctuation, add translation output
                        if not (translation_output in ".,?!:;" and result and result[-1] == translation_output):
                            result.append(translation_output)
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

    # Only apply voice translations for Vosk (Whisper handles punctuation natively)
    processed = process_voice_translations(words) if ENGINE == "vosk" else words

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
    global whisper_model, last_char_typed

    pipeline_start = time.perf_counter()

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

        recording_done = time.perf_counter()

        # Transcribe accumulated audio when key is released
        if audio_chunks:
            before_concat = time.perf_counter()
            audio_data = np.concatenate(audio_chunks)
            after_concat = time.perf_counter()

            # Skip if recording is too short (less than 0.5 seconds)
            if len(audio_data) >= SAMPLE_RATE * 0.5:
                # Time the transcription
                before_transcribe = time.perf_counter()
                timestamp = datetime.now()

                start_time = time.perf_counter()

                # Use optimized settings for faster transcription
                segments, _ = whisper_model.transcribe(
                    audio_data,
                    language="en",
                    beam_size=1,                        # Reduced from default 5
                    temperature=0.0,                    # Disable fallback cascade
                    condition_on_previous_text=False,   # Not needed for short recordings
                    vad_filter=False,                   # User controls start/stop
                    # compression_ratio_threshold uses default 2.4 to prevent hallucinations
                )

                # Force evaluation of segments (generator) and collect text
                # This is where the actual transcription work happens!
                text = " ".join(segment.text.strip() for segment in segments)

                after_transcribe = time.perf_counter()
                elapsed_ms = (after_transcribe - start_time) * 1000

                if text:
                    # Clean up Whisper quirks
                    before_cleanup = time.perf_counter()
                    text = text.replace(",,", ",")  # Multiple commas to single comma
                    if text.endswith("..."):
                        text = text[:-3]  # Remove trailing ellipsis
                    after_cleanup = time.perf_counter()

                    # Split words and separate punctuation for voice translation matching
                    before_tokenize = time.perf_counter()
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
                    after_tokenize = time.perf_counter()

                    if tokens:
                        typing_start = time.perf_counter()
                        type_text(tokens)
                        # Add space after transcribed text only if it doesn't end with punctuation
                        # and we didn't just type a space (prevent double spaces)
                        last_token = tokens[-1]
                        if last_token not in (".", ",", "?", "!", ":", ";") and last_char_typed != " ":
                            kb_controller.type(" ")
                            last_char_typed = " "
                        typing_done = time.perf_counter()

                        # Calculate timing breakdown - EVERYTHING from key release to typing done
                        user_latency_ms = (typing_done - key_release_time) * 1000 if key_release_time else 0

                        # Detailed breakdown of every stage
                        wait_stop_ms = (recording_done - key_release_time) * 1000 if key_release_time else 0
                        gap1_ms = (before_concat - recording_done) * 1000
                        concat_ms = (after_concat - before_concat) * 1000
                        gap2_ms = (before_transcribe - after_concat) * 1000
                        transcribe_ms = elapsed_ms  # Now includes text collection (generator evaluation)
                        cleanup_ms = (after_cleanup - before_cleanup) * 1000
                        tokenize_ms = (after_tokenize - before_tokenize) * 1000
                        gap3_ms = (typing_start - after_tokenize) * 1000
                        typing_ms = (typing_done - typing_start) * 1000

                        # Sum check
                        sum_ms = (wait_stop_ms + gap1_ms + concat_ms + gap2_ms + transcribe_ms +
                                  cleanup_ms + tokenize_ms + gap3_ms + typing_ms)

                        # Log: timestamp | text | breakdown
                        print(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} | {text}")
                        print(f"  LATENCY={user_latency_ms:.0f}ms SUM={sum_ms:.0f}ms")
                        print(f"  wait_stop={wait_stop_ms:.0f}ms + concat={concat_ms:.0f}ms + "
                              f"transcribe={transcribe_ms:.0f}ms + cleanup={cleanup_ms:.0f}ms + "
                              f"tokenize={tokenize_ms:.0f}ms + typing={typing_ms:.0f}ms")

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
    global is_recording, recording_thread, currently_pressed_keys, key_release_time

    # Remove from pressed keys
    currently_pressed_keys.discard(key)

    # If we were recording and any trigger key was released, stop recording
    if is_recording and key in TRIGGER_KEYS:
        with lock:
            if is_recording:
                key_release_time = time.perf_counter()  # Track when user released key
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

Available keys: """ + ", ".join(sorted(k for k in KEY_MAP.keys() if not k.startswith('ctl_'))) + """

Key combinations: Use '-' to combine keys (e.g., shift_l-ctrl_l, shift_l-alt_r)
"""
    )

    parser.add_argument("--engine", choices=["whisper", "vosk"], required=True,
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
        # Get model from config based on engine
        config_key = f"{engine}-model"
        model_name = config.get(config_key)
        if not model_name:
            print(f"Error: No model specified for {engine} engine")
            print(f"Either set '{config_key}' in config.yaml or use --model on command line")
            return 1
        model_name = resolve_model_name(model_name, engine)

    # Parse trigger key - support combinations like "shift-ctrl_r"
    if "-" in args.key:
        key_parts = [k.strip() for k in args.key.split("-")]
        TRIGGER_KEYS = set()
        for k in key_parts:
            key_obj = KEY_MAP.get(k, KEY_MAP.get(k.lower()))
            if key_obj is None:
                # Show documented keys only (exclude ctl_ aliases)
                valid_keys = sorted(k for k in KEY_MAP.keys() if not k.startswith('ctl_'))
                print(f"Error: Invalid key name '{k}'")
                print(f"Valid keys: {', '.join(valid_keys)}")
                return 1
            TRIGGER_KEYS.add(key_obj)
    else:
        key_obj = KEY_MAP.get(args.key, KEY_MAP.get(args.key.lower()))
        if key_obj is None:
            # Show documented keys only (exclude ctl_ aliases)
            valid_keys = sorted(k for k in KEY_MAP.keys() if not k.startswith('ctl_'))
            print(f"Error: Invalid key name '{args.key}'")
            print(f"Valid keys: {', '.join(valid_keys)}")
            return 1
        TRIGGER_KEYS = {key_obj}
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
            compute_type="int8",
            cpu_threads=4,  # Prevents thread contention
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

voice2keyboard
==============

Voice to Keyboard Typing


## Synopsis

```bash
make run engine=whisper key=alt_r
make run engine=vosk key=shift_l-ctrl_r
make run engine=whisper key=ctrl_l model=medium.en
```


## Description

`voice2keyboard` is a Linux program that provides real-time voice-to-text
input.
Hold a trigger key (or key combination) to record, and words appear at your
cursor as you speak.

Supports two speech recognition engines:
- **Vosk** - Fast, offline, streaming recognition
- **Whisper** - Higher accuracy, better with technical terms, batch processing

Can be set up as a systemd user service for always-on availability.


## Features

- **Dual engine support** - Choose between Vosk (streaming) or Whisper
  (accurate)
- **Push-to-talk input** - Hold a trigger key or combination (e.g., Shift+Ctrl)
  to activate recording
- **Two typing modes** - Buffered (accurate, waits for pauses) or Realtime
  (immediate)
- **Works everywhere** - Types into any application that accepts keyboard input
- **Voice translations** - Say "tadpole", "question mark", etc. to insert
  punctuation
- **Offline processing** - Both engines work locally, no internet required
- **Auto-start on login** - Can run as a systemd user service
- **Configurable** - Customize engine, trigger key, typing mode, and voice
  translations via YAML


## Requirements


### System

- Linux (tested on Ubuntu/Debian)
- X11 display server (Wayland support is limited)
- Working microphone


### Dependencies

**Manual install required:**
```bash
sudo apt install alsa-utils   # provides arecord
```

**Automatically managed by Makefile:**
- Python 3.13.1 (downloaded to `.cache/`)
- pynput (hotkey detection and keyboard simulation)
- PyYAML (configuration parsing)
- vosk (for Vosk engine)
- faster-whisper (for Whisper engine)
- Speech models (downloaded based on engine and config)


## Installation


### Quick Start

```bash
# Clone the repository
git clone https://github.com/ingydotnet/voice2keyboard
cd voice2keyboard

# Install and start the service
make install
```

That's it! The service will:
1. Download Python 3.13.1 and create a virtual environment
2. Install required Python packages
3. Download the configured speech recognition model
4. Install and enable the systemd user service
5. Start the service immediately


### Verify Installation

```bash
make status   # Check if service is running
make logs     # View live logs
```

## Usage

1. **Hold the trigger key** (default: Right Alt) in any application
2. **Speak** - words appear at your cursor
3. **Release the key** to stop recording


### Typing Modes

**Buffered mode (default, recommended):**
- Waits for natural pauses before typing
- More accurate word recognition
- Optional pause delay for better context

**Realtime mode:**
- Types words immediately as recognized
- Lower latency but may misrecognize
- Only works with Vosk engine


### Voice Translations

Say these words to insert punctuation and special characters:

| Say | Types |
|-----|-------|
| "full stop" / "full step" | `.` |
| "tadpole" | `,` |
| "question mark" | `?` |
| "exclamation mark" | `!` |
| "colon" | `:` |
| "semicolon" | `;` |
| "hyphen" / "dash" | `-` |
| "quote" | `"` |
| "return" | newline |
| "paragraph" | double newline |
| "zero" - "ten" | `0` - `10` |

Voice translations are case-insensitive and work even when Whisper adds its own
punctuation.


## Configuration

Edit `config.yaml` to customize behavior:

**Note:** Engine and trigger key are specified via command-line arguments, not in
config.

```yaml
# Model names for each engine (can override with --model on command line)
# Vosk models: vosk-model-small-en-us-0.15, vosk-model-en-us-0.22-lgraph
# Whisper models: tiny.en, base.en, small.en, medium.en
vosk-model: vosk-model-small-en-us-0.15
whisper-model: small.en

# Typing mode
# - buffered: Wait for natural pauses before typing (more accurate)
# - realtime: Type words immediately as recognized (faster, less accurate)
mode: buffered

# Pause delay (in seconds) after speech pause before typing
# Gives the model extra time to refine predictions
# Set to 0 to disable. Only applies in buffered mode.
pause: 0.3

# Voice translations - say the word, get the symbol
translations:
  full stop: "."
  tadpole: ","
  question mark: "?"
  # Add your own...
```


## Command Line Usage

The `--engine` and `--key` parameters are required for all manual runs.
Model selection uses the engine-specific config value (`vosk-model` or
`whisper-model`) unless overridden with `--model`.


### Running Manually

```bash
# Run with Whisper engine and Alt_R key
make run engine=whisper key=alt_r

# Run with Vosk engine and key combination
make run engine=vosk key=shift_l-ctrl_r

# With additional options
make run engine=whisper key=ctrl_l mode=buffered

# Override model
make run engine=whisper key=alt_r model=medium.en

# Multiple options
make run engine=vosk key=shift_l-alt_r mode=realtime pause=0.5
```


### Direct Python Invocation

```bash
# Basic usage (engine and key required)
python voice2keyboard.py --engine whisper --key alt_r

# With options
python voice2keyboard.py --engine vosk --key shift_l-alt_r --mode buffered

# Key combination
python voice2keyboard.py --engine whisper --key shift_l-ctrl_r

# See all options
python voice2keyboard.py --help
```


### Performance Logging

Track transcription performance and identify bottlenecks:

```bash
# Log to file (no console output)
make run engine=whisper key=alt_r log=performance.txt

# View the log in another terminal
tail -f performance.txt
```

**Log output format:**
```
2025-12-26 00:09:28 | Hopefully this is still fast.
  LATENCY=1080ms SUM=1080ms
  wait_stop=104ms + concat=0ms + transcribe=926ms + cleanup=0ms + tokenize=0ms + typing=49ms
```

**Timing breakdown:**
- **LATENCY**: Total user-perceived delay (key release → text appears)
- **wait_stop**: Time for recording loop to stop after key release
- **concat**: Audio concatenation time
- **transcribe**: Whisper transcription time (includes text collection)
- **cleanup**: Text cleanup time (removing Whisper artifacts)
- **tokenize**: Token processing time
- **typing**: Keyboard simulation time

Use this to:
- Identify performance bottlenecks
- Compare different models
- Test optimization settings
- Debug latency issues


## Running Multiple Instances

You can run multiple instances simultaneously with different engines and
trigger keys:

**Example:**

```bash
# Terminal 1 - Whisper for accuracy
make run engine=whisper key=alt_r

# Terminal 2 - Vosk for speed
make run engine=vosk key=ctrl_l
```

Now you can use `Alt_R` for accurate Whisper transcription and `Ctrl_L` for
fast Vosk transcription!

**Important considerations:**

1. **Use non-overlapping trigger keys**
   - ✅ Good: `alt_r` and `ctrl_l`
   - ✅ Good: `alt_r` and `shift_l-ctrl_r`
   - ❌ Bad: `alt_r` and `shift_l-alt_r` (both include `alt_r`)

2. **Microphone access**
   - Most Linux systems handle multiple processes accessing the mic
     simultaneously
   - Avoid holding both trigger keys at the same time
   - Best practice: use one instance at a time

3. **Use cases**
   - Quick switching: Whisper for technical writing, Vosk for casual notes
   - Different contexts: One key per hand, or different keys for different
     postures
   - A/B testing: Compare engine accuracy side-by-side

4. **Systemd service**
   - The default service runs a single instance
   - For multiple persistent instances, create separate service files
   - Or run additional instances manually in terminals


## Speech Engines


### Vosk

**Pros:**
- Streaming recognition (types as you speak in realtime mode)
- Lower latency
- Smaller models

**Cons:**
- Lower accuracy
- Less familiar with technical terms

**Models:**
- Download from [alphacephei.com/vosk/models](
  https://alphacephei.com/vosk/models)
- Extracted to project root directory

| Model | Size | Notes |
|-------|------|-------|
| `vosk-model-small-en-us-0.15` | ~40MB | Default, fast |
| `vosk-model-en-us-0.22-lgraph` | ~128MB | Better accuracy |
| `vosk-model-en-us-0.22` | ~1.8GB | Best accuracy |


### Whisper

**Pros:**
- Higher accuracy
- Better with technical vocabulary (CLI, GitHub, API, etc.)
- Handles punctuation naturally

**Cons:**
- Batch processing only (no realtime mode)
- Higher latency
- Larger models

**Models:**
- Downloaded automatically via faster-whisper
- Cached in `~/.cache/huggingface/`

| Model | Size | Notes |
|-------|------|-------|
| `tiny.en` | ~75MB | Fastest, lowest accuracy |
| `base.en` | ~140MB | Good balance |
| `small.en` | ~466MB | **Recommended** - good speed/accuracy |
| `medium.en` | ~1.5GB | Best accuracy, slower |


## Key Combinations

You can use modifier key combinations as triggers:

**Examples:**
- `shift_l-ctrl_l` - Left Shift + Left Ctrl
- `shift_l-ctrl_r` - Left Shift + Right Ctrl
- `shift_l-alt_r` - Left Shift + Right Alt
- `ctrl_l-alt_l-shift_l` - All three left modifiers

**Available keys:**
- Modifiers: `alt_l`, `alt_r`, `ctrl_l`, `ctrl_r`, `shift_l`, `shift_r`
- Special: `scroll_lock`, `pause`, `insert`, `delete`

**Note:** Order doesn't matter - `ctrl_r-shift_l` and `shift_l-ctrl_r` are
equivalent.


## Make Commands

| Command | Description |
|---------|-------------|
| `make install` | Install as systemd service (auto-starts on login) |
| `make uninstall` | Stop and remove the service |
| `make status` | Check service status |
| `make logs` | View live logs (journalctl) |
| `make run engine=ENGINE` | Run manually (engine required: vosk or whisper) |
| `make help` | Show voice2keyboard help |
| `make clean` | Remove generated files |
| `make realclean` | Remove everything including models |


## Architecture

```
voice2keyboard/
├── voice2keyboard.py      # Main daemon script (~500 lines)
├── config.yaml            # Configuration file
├── voice2keyboard.service # systemd service template
├── Makefile               # Build system (uses 'makes' framework)
├── CLAUDE.md              # AI assistant instructions
└── .cache/                # Auto-generated: Python, venv, build tools
```


### How It Works

1. **Hotkey detection** - pynput monitors keyboard events for the trigger
   key/combination
2. **Audio capture** - arecord captures microphone input at 16kHz mono
3. **Speech recognition**:
   - **Vosk**: Streams audio for real-time processing
   - **Whisper**: Accumulates audio, transcribes on key release
4. **Voice translation processing** - Converts special words to punctuation
5. **Keyboard simulation** - pynput types recognized text at cursor position


### Processing Pipeline

**Vosk (Buffered Mode):**
```
Hold key → 
Stream audio → 
Detect pause → 
Type complete phrase → 
Release key
```

**Vosk (Realtime Mode):**
```
Hold key → 
Stream audio → 
Type words immediately → 
Release key
```

**Whisper (Buffered Mode):**
```
Hold key → 
Accumulate audio → 
Release key → 
Transcribe → 
Process voice translations → 
Type text
```


## Troubleshooting


### Service won't start

```bash
make logs                           # Check for errors
systemctl --user status voice2keyboard
```


### No audio input

```bash
arecord -d 3 test.wav && aplay test.wav   # Test microphone
```


### Permission errors with keyboard

On some systems, you may need to add your user to the `input` group:
```bash
sudo usermod -a -G input $USER
# Log out and back in
```


### X11 vs Wayland

pynput works best with X11.
If using Wayland, you may need to run under XWayland or switch to X11 session.

Check your session type:
```bash
echo $XDG_SESSION_TYPE
```


### Model not found

**Vosk:** Manually download from [alphacephei.com/vosk/models](
https://alphacephei.com/vosk/models) and extract to project root.

**Whisper:** Check internet connection - models download automatically on first
use.


### Double spaces or punctuation issues

Make sure you're using the latest version - voice translation processing has been
improved to handle Whisper's automatic punctuation.


### Engine not installed

```bash
# If you get "Vosk engine selected but not installed"
pip install vosk

# If you get "Whisper engine selected but not installed"
pip install faster-whisper numpy
```


## Development


### Running from source

```bash
make run engine=whisper key=alt_r   # Uses managed Python environment
```


### Project dependencies

The Makefile uses the [makes](https://github.com/makeplus/makes) framework
which:
- Downloads and manages a standalone Python 3.13.1 installation
- Creates an isolated virtual environment
- Handles model downloading and extraction
- Generates the systemd service file with correct paths


### Model name matching

Model names support regex patterns:
```bash
make run engine=whisper key=alt_r model=small  # Matches small.en
make run engine=vosk key=alt_r model=small     # Matches vosk-model-small-en-us-0.15
```


## Tips


### Choosing an Engine

**Use Vosk if:**
- You want low latency
- You prefer realtime typing
- You have limited disk space

**Use Whisper if:**
- You need high accuracy
- You frequently use technical terms
- You can tolerate slight delay


### Choosing a Model

**Vosk:**
- Start with `vosk-model-small-en-us-0.15`
- Upgrade to `vosk-model-en-us-0.22-lgraph` for better accuracy

**Whisper:**
- Use `small.en` for best speed/accuracy balance
- Try `base.en` on slower machines
- Use `medium.en` for maximum accuracy (slower)


### Optimizing Accuracy

1. Speak clearly and at normal pace
2. Use buffered mode with pause delay (default 0.3s)
3. Add common misrecognitions to voice translations
4. Use Whisper for technical content


## License

MIT License - see [License](License) for details.


## Contributing

Contributions welcome! Please open an issue or pull request.

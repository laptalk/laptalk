M := .cache/makes
$(shell [ -d $M ] || ( git clone -q https://github.com/makeplus/makes $M))

# Use Python 3.13 (3.14 has compatibility issues with faster-whisper)
PYTHON-VERSION ?= 3.13.1

include $M/init.mk
include $M/python.mk
include $M/clean.mk
include $M/shell.mk
include $M/agents.mk

CONFIG_ENGINE := $(shell grep ^engine: config.yaml | cut -f2 -d' ' | tr -d ' ')
CONFIG_ENGINE ?= vosk

MODEL := $(shell grep ^model: config.yaml | cut -f2 -d' ')
MODEL ?= vosk-model-small-en-us-0.15

# Default Whisper model if not specified
WHISPER_MODEL ?= small.en

VOSK-URL := https://alphacephei.com/vosk/models

MAKES-REALCLEAN := \
	vosk-* \
	.whisper-model-downloaded \
	__pycache__/ \

# Base dependencies (Python and packages)
DEPS := \
  $(PYTHON) \
  $(PYTHON-VENV)/bin/pynput \

# Additional dependencies based on engine (for install target)
ifeq ($(CONFIG_ENGINE),whisper)
  INSTALL_DEPS := $(DEPS) $(PYTHON-VENV)/bin/faster_whisper .whisper-model-downloaded
else
  INSTALL_DEPS := $(DEPS) $(MODEL)
endif

SERVICE-FILE := $(HOME)/.config/systemd/user/voice2keyboard.service

export XDG_SESSION_TYPE := x11

# Make variables for run target (override from command line)
engine ?=
model ?=
key ?=
mode ?=
pause ?=

# Build command-line arguments
RUN_ARGS :=
ifneq ($(engine),)
  RUN_ARGS += --engine=$(engine)
endif
ifneq ($(model),)
  RUN_ARGS += --model=$(model)
endif
ifneq ($(key),)
  RUN_ARGS += --key=$(key)
endif
ifneq ($(mode),)
  RUN_ARGS += --mode=$(mode)
endif
ifneq ($(pause),)
  RUN_ARGS += --pause=$(pause)
endif

run: $(DEPS)
ifeq ($(engine),)
	@echo "Error: engine parameter required"
	@echo "Usage: make run engine=whisper key=alt_r"
	@echo "       make run engine=vosk key=shift_l-ctrl_r"
	@exit 1
endif
ifeq ($(key),)
	@echo "Error: key parameter required"
	@echo "Usage: make run engine=whisper key=alt_r"
	@echo "       make run engine=vosk key=shift_l-ctrl_r"
	@exit 1
endif
ifeq ($(engine),vosk)
	@$(MAKE) -s $(MODEL)
endif
ifeq ($(engine),whisper)
	@$(MAKE) -s $(PYTHON-VENV)/bin/faster_whisper .whisper-model-downloaded
endif
	python voice2keyboard.py $(RUN_ARGS)

help: $(DEPS)
	python voice2keyboard.py --help

install: $(INSTALL_DEPS) $(SERVICE-FILE)
	systemctl --user daemon-reload
	systemctl --user enable voice2keyboard
	systemctl --user start voice2keyboard
	@echo "voice2keyboard installed and running"
	@echo "Hold Delete key to record and type"

$(SERVICE-FILE): voice2keyboard.service
	mkdir -p $(dir $@)
	sed 's|%h/src/voice2keyboard|$(ROOT)|g' $< > $@

uninstall:
	-systemctl --user stop voice2keyboard
	-systemctl --user disable voice2keyboard
	rm -f $(SERVICE-FILE)
	systemctl --user daemon-reload
	@echo "voice2keyboard uninstalled"

status:
	systemctl --user status voice2keyboard

logs:
	journalctl --user -u voice2keyboard -f

$(PYTHON-VENV)/bin/pynput: $(PYTHON-VENV)
	pip install -q pynput pyyaml vosk

$(PYTHON-VENV)/bin/faster_whisper: $(PYTHON-VENV)
	pip install -q faster-whisper numpy

.whisper-model-downloaded: $(PYTHON-VENV)/bin/faster_whisper
	python -c "from faster_whisper import WhisperModel; WhisperModel('$(if $(model),$(model),$(WHISPER_MODEL))')"
	touch $@

$(MODEL): $(MODEL).zip
	unzip $<
	touch $@

$(MODEL).zip:
	wget $(VOSK-URL)/$@

# Parakeet AI Prototype

A lightweight desktop application that records microphone audio, transcribes speech locally using Whisper, and streams AI-suggested answers using a local Ollama model.

---

## Features

- **Microphone Capture**: Records audio directly from your default input device using `sounddevice`.
- **Local Speech-to-Text**: Fast offline transcription using `faster-whisper` (`base.en`).
- **Progressive Streaming**: Real-time token-by-token answer generation powered by a local Ollama model (`llama3.2:1b`).
- **Clean UI**: Native desktop interface built with Python's built-in `tkinter`.

---

## Prerequisites

1. **Python 3.10+** installed on your system.
2. **Ollama** installed and running:
   - Download from [ollama.com](https://ollama.com).
   - Pull the required model:
     ```bash
     ollama pull llama3.2:1b
     ```
   - Ensure the Ollama service is running in the background.

---

## Installation

1. Clone or download this repository:
   ```bash
   git clone <repo-url>
   cd <repo-folder>
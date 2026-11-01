import os
import tempfile
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
import ollama


class ParakeetApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Parakeet AI Prototype")
        self.root.geometry("600x550")

        self.sample_rate = 16000
        self.is_recording = False
        self.audio_frames = []

        self.btn_frame = ttk.Frame(self.root, padding=10)
        self.btn_frame.pack(fill=tk.X)

        self.btn_start = ttk.Button(self.btn_frame, text="Start Recording", command=self.start_recording)
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.btn_stop = ttk.Button(self.btn_frame, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        self.btn_suggest = ttk.Button(self.btn_frame, text="Suggest Answer", command=self.suggest_answer, state=tk.DISABLED)
        self.btn_suggest.pack(side=tk.LEFT, padx=5)

        self.status_label = ttk.Label(self.root, text="Loading Whisper model...", font=("Arial", 9))
        self.status_label.pack(anchor=tk.W, padx=10)

        lbl_t = ttk.Label(self.root, text="Transcription:", font=("Arial", 10, "bold"))
        lbl_t.pack(anchor=tk.W, padx=10, pady=(10, 2))

        self.transcription_box = scrolledtext.ScrolledText(self.root, height=6, wrap=tk.WORD)
        self.transcription_box.pack(fill=tk.X, padx=10, pady=2)

        lbl_a = ttk.Label(self.root, text="AI Suggested Answer:", font=("Arial", 10, "bold"))
        lbl_a.pack(anchor=tk.W, padx=10, pady=(10, 2))

        self.answer_box = scrolledtext.ScrolledText(self.root, height=12, wrap=tk.WORD)
        self.answer_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        t = threading.Thread(target=self.load_model)
        t.start()

    def update_status(self, text):
        self.root.after(0, lambda: self.status_label.config(text=f"Status: {text}"))

    def load_model(self):
        self.model = WhisperModel("base.en", device="cpu", compute_type="int8")
        self.update_status("Ready")

    def start_recording(self):
        self.audio_frames = []
        self.is_recording = True

        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_suggest.config(state=tk.DISABLED)

        self.transcription_box.delete("1.0", tk.END)
        self.answer_box.delete("1.0", tk.END)
        self.update_status("Recording audio...")

        def record_callback(indata, frames, time, status):
            if self.is_recording:
                self.audio_frames.append(indata.copy())

        self.stream = sd.InputStream(samplerate=self.sample_rate, channels=1, callback=record_callback)
        self.stream.start()

    def stop_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False
        self.stream.stop()
        self.stream.close()

        self.btn_stop.config(state=tk.DISABLED)
        self.btn_start.config(state=tk.NORMAL)
        self.update_status("Processing transcription...")

        threading.Thread(target=self.run_whisper).start()

    def run_whisper(self):
        if len(self.audio_frames) == 0:
            self.update_status("No audio found")
            return

        audio = np.concatenate(self.audio_frames, axis=0).flatten()

        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

        temp_file = tempfile.mktemp(suffix=".wav")
        sf.write(temp_file, audio, self.sample_rate)

        try:
            segments, _ = self.model.transcribe(temp_file, beam_size=3)
            result = ""
            for seg in segments:
                result += seg.text + " "
            result = result.strip()

            def show_text():
                self.transcription_box.insert(tk.END, result)
                if len(result) > 0:
                    self.btn_suggest.config(state=tk.NORMAL)
                self.update_status("Transcription done")

            self.root.after(0, show_text)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def suggest_answer(self):
        prompt = self.transcription_box.get("1.0", tk.END).strip()
        if not prompt:
            return

        self.btn_suggest.config(state=tk.DISABLED)
        self.btn_start.config(state=tk.DISABLED)
        self.answer_box.delete("1.0", tk.END)
        self.update_status("Generating response...")

        threading.Thread(target=self.run_ollama, args=(prompt,)).start()

    def run_ollama(self, prompt):
        try:
            response = ollama.chat(
                model="llama3.2:1b",
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )

            for chunk in response:
                word = chunk["message"]["content"]
                self.root.after(0, self.add_token, word)

        except Exception as e:
            self.root.after(0, self.add_token, f"\nError: {e}")
        finally:
            self.root.after(0, self.done_generating)

    def add_token(self, token):
        self.answer_box.insert(tk.END, token)
        self.answer_box.see(tk.END)

    def done_generating(self):
        self.btn_suggest.config(state=tk.NORMAL)
        self.btn_start.config(state=tk.NORMAL)
        self.update_status("Ready")


if __name__ == "__main__":
    win = tk.Tk()
    app = ParakeetApp(win)
    win.mainloop()
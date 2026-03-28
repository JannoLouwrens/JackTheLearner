"""
AUDIO LISTENER - Microphone Input for Jack

Captures audio from the microphone, provides it to the brain as:
1. Raw waveform tensor → AudioEncoder (ambient sound understanding)
2. Transcribed text → Language system (voice commands)

Two transcription modes:
- Local Whisper (via AudioEncoder.transcribe) - offline, slower
- API-based (OpenAI Whisper API or local faster-whisper) - faster

Architecture:
    Microphone → background thread → circular buffer
                                       ├→ Raw waveform [1, samples] → brain.forward(audio=...)
                                       └→ Whisper transcription → brain.chat(text) / TaskManager

Research:
- Whisper (OpenAI 2022): Robust speech recognition
- VAD (Voice Activity Detection): Only process when someone is speaking

Author: Janno Louwrens
"""

import time
import threading
import numpy as np
from typing import Optional, Callable
from collections import deque
from dataclasses import dataclass

try:
    import torch
except ImportError:
    torch = None


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class AudioConfig:
    """Audio listener configuration."""
    sample_rate: int = 16000          # 16kHz (Whisper standard)
    chunk_duration: float = 0.1       # 100ms chunks from mic
    buffer_seconds: float = 5.0       # Keep last 5 seconds of audio
    silence_threshold: float = 0.01   # RMS below this = silence
    speech_min_duration: float = 0.5  # Minimum speech length to process
    speech_max_duration: float = 10.0 # Maximum speech length
    silence_after_speech: float = 0.8 # Silence duration to end speech
    transcription_mode: str = "local" # "local" (Whisper) or "api" (OpenAI API)


# ==============================================================================
# AUDIO LISTENER
# ==============================================================================

class AudioListener:
    """
    Listens to the microphone and provides audio data to the brain.

    Runs a background thread that:
    1. Captures audio chunks from the microphone
    2. Detects when someone is speaking (VAD)
    3. When speech ends, transcribes it
    4. Provides raw waveform for the AudioEncoder

    Usage:
        listener = AudioListener(config)
        listener.start()

        # In game loop:
        waveform = listener.get_recent_audio()  # For AudioEncoder
        text = listener.get_transcription()       # For language system

        listener.stop()
    """

    def __init__(self, config: AudioConfig = None, on_transcription: Callable = None):
        """
        Args:
            config: Audio configuration
            on_transcription: Callback when speech is transcribed.
                              Called with (text: str) from the background thread.
        """
        self.config = config or AudioConfig()
        self.on_transcription = on_transcription

        # Audio buffer (circular)
        buffer_samples = int(self.config.buffer_seconds * self.config.sample_rate)
        self._buffer = np.zeros(buffer_samples, dtype=np.float32)
        self._buffer_pos = 0
        self._lock = threading.Lock()

        # Speech detection state
        self._is_speaking = False
        self._speech_start = 0.0
        self._silence_start = 0.0
        self._speech_buffer = []

        # Transcription results
        self._last_transcription = ""
        self._transcription_ready = False

        # Thread control
        self._running = False
        self._thread = None
        self._available = False
        self._stream = None

        # Check if audio capture is available
        try:
            import sounddevice as sd
            self._sd = sd
            self._available = True
            print("[AUDIO] sounddevice available - microphone ready")
        except ImportError:
            try:
                import pyaudio
                self._pyaudio = pyaudio
                self._available = True
                print("[AUDIO] pyaudio available - microphone ready")
            except ImportError:
                print("[AUDIO] No audio library (pip install sounddevice or pyaudio)")
                self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def start(self):
        """Start listening to the microphone."""
        if not self._available:
            print("[AUDIO] Cannot start - no audio library")
            return

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[AUDIO] Listening at {self.config.sample_rate}Hz")

    def stop(self):
        """Stop listening."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        print("[AUDIO] Stopped")

    def get_recent_audio(self, duration: float = 1.0) -> Optional['torch.Tensor']:
        """
        Get the most recent audio as a tensor for the AudioEncoder.

        Args:
            duration: How many seconds of recent audio to return

        Returns:
            Tensor [1, samples] at 16kHz, or None if not available
        """
        if torch is None or not self._available:
            return None

        samples = int(duration * self.config.sample_rate)
        with self._lock:
            # Extract from circular buffer
            end = self._buffer_pos
            start = end - samples
            if start < 0:
                # Wrap around
                audio = np.concatenate([self._buffer[start:], self._buffer[:end]])
            else:
                audio = self._buffer[start:end].copy()

        if len(audio) == 0:
            return None

        return torch.from_numpy(audio).float().unsqueeze(0)  # [1, samples]

    def get_transcription(self) -> Optional[str]:
        """
        Get the latest transcription if available.

        Returns the transcribed text and clears it (consume-once pattern).
        Returns None if no new transcription.
        """
        if self._transcription_ready:
            self._transcription_ready = False
            text = self._last_transcription
            self._last_transcription = ""
            return text
        return None

    def is_speaking(self) -> bool:
        """Check if someone is currently speaking."""
        return self._is_speaking

    def get_ambient_level(self) -> float:
        """Get current ambient sound level (RMS, 0-1 scale).
        Useful for detecting loud events (something fell, door slam, etc.)."""
        with self._lock:
            # Check last chunk's RMS
            recent = self._buffer[max(0, self._buffer_pos - 1600):self._buffer_pos]
            if len(recent) == 0:
                return 0.0
            return float(min(1.0, np.sqrt(np.mean(recent ** 2)) * 10))

    # ─────────────────────────────────────────────────────────────────────
    # BACKGROUND THREAD
    # ─────────────────────────────────────────────────────────────────────

    def _capture_loop(self):
        """Background thread: capture audio from microphone."""
        chunk_samples = int(self.config.chunk_duration * self.config.sample_rate)

        if hasattr(self, '_sd'):
            self._capture_sounddevice(chunk_samples)
        elif hasattr(self, '_pyaudio'):
            self._capture_pyaudio(chunk_samples)

    def _capture_sounddevice(self, chunk_samples: int):
        """Capture using sounddevice library."""
        sd = self._sd
        try:
            with sd.InputStream(samplerate=self.config.sample_rate,
                                channels=1, dtype='float32',
                                blocksize=chunk_samples) as stream:
                while self._running:
                    data, overflowed = stream.read(chunk_samples)
                    if overflowed:
                        continue
                    chunk = data.flatten()
                    self._process_chunk(chunk)
        except Exception as e:
            print(f"[AUDIO] Capture error: {e}")
            self._running = False

    def _capture_pyaudio(self, chunk_samples: int):
        """Capture using pyaudio library."""
        pa = self._pyaudio
        try:
            p = pa.PyAudio()
            stream = p.open(
                format=pa.paFloat32,
                channels=1,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=chunk_samples,
            )
            while self._running:
                data = stream.read(chunk_samples, exception_on_overflow=False)
                chunk = np.frombuffer(data, dtype=np.float32)
                self._process_chunk(chunk)
            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception as e:
            print(f"[AUDIO] Capture error: {e}")
            self._running = False

    def _process_chunk(self, chunk: np.ndarray):
        """Process an audio chunk: buffer it and detect speech."""
        # Write to circular buffer
        with self._lock:
            n = len(chunk)
            end = self._buffer_pos + n
            buf_len = len(self._buffer)
            if end <= buf_len:
                self._buffer[self._buffer_pos:end] = chunk
            else:
                first = buf_len - self._buffer_pos
                self._buffer[self._buffer_pos:] = chunk[:first]
                self._buffer[:n - first] = chunk[first:]
            self._buffer_pos = end % buf_len

        # Voice Activity Detection (simple energy-based)
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        now = time.monotonic()

        if rms > self.config.silence_threshold:
            # Sound detected
            if not self._is_speaking:
                self._is_speaking = True
                self._speech_start = now
                self._speech_buffer = []
            self._silence_start = 0.0
            self._speech_buffer.append(chunk.copy())
        else:
            # Silence
            if self._is_speaking:
                if self._silence_start == 0.0:
                    self._silence_start = now
                    self._speech_buffer.append(chunk.copy())
                elif now - self._silence_start > self.config.silence_after_speech:
                    # Speech ended - process it
                    speech_duration = now - self._speech_start
                    if speech_duration >= self.config.speech_min_duration:
                        self._on_speech_end()
                    self._is_speaking = False
                    self._speech_buffer = []
                    self._silence_start = 0.0
                else:
                    self._speech_buffer.append(chunk.copy())

            # Limit speech duration
            if self._is_speaking and now - self._speech_start > self.config.speech_max_duration:
                self._on_speech_end()
                self._is_speaking = False
                self._speech_buffer = []

    def _on_speech_end(self):
        """Called when a speech segment is detected. Transcribe it."""
        if not self._speech_buffer:
            return

        speech = np.concatenate(self._speech_buffer)

        # Transcribe in background (don't block capture)
        threading.Thread(
            target=self._transcribe,
            args=(speech,),
            daemon=True,
        ).start()

    def _transcribe(self, speech: np.ndarray):
        """Transcribe speech audio to text."""
        text = ""

        if self.config.transcription_mode == "api":
            text = self._transcribe_api(speech)
        else:
            text = self._transcribe_local(speech)

        if text and text.strip():
            text = text.strip()
            self._last_transcription = text
            self._transcription_ready = True
            print(f"[AUDIO] Heard: \"{text}\"")

            # Call callback if set
            if self.on_transcription is not None:
                try:
                    self.on_transcription(text)
                except Exception as e:
                    print(f"[AUDIO] Callback error: {e}")

    def _transcribe_local(self, speech: np.ndarray) -> str:
        """Transcribe using local Whisper (via transformers or faster-whisper)."""
        try:
            # Try faster-whisper first (much faster)
            from faster_whisper import WhisperModel
            if not hasattr(self, '_whisper_model'):
                self._whisper_model = WhisperModel("tiny", compute_type="int8")
            segments, _ = self._whisper_model.transcribe(speech)
            return " ".join(s.text for s in segments)
        except ImportError:
            pass

        try:
            # Fall back to transformers Whisper
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            if not hasattr(self, '_whisper_processor'):
                self._whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
                self._whisper_hf_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
            inputs = self._whisper_processor(speech, sampling_rate=self.config.sample_rate, return_tensors="pt")
            ids = self._whisper_hf_model.generate(inputs.input_features)
            return self._whisper_processor.batch_decode(ids, skip_special_tokens=True)[0]
        except ImportError:
            pass

        return "[No Whisper available - install faster-whisper or transformers]"

    def _transcribe_api(self, speech: np.ndarray) -> str:
        """Transcribe using OpenAI Whisper API."""
        try:
            import openai
            import tempfile
            import wave
            import os

            # Save as WAV temporarily
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
                with wave.open(f, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(self.config.sample_rate)
                    wf.writeframes((speech * 32767).astype(np.int16).tobytes())

            client = openai.OpenAI()
            with open(tmp_path, "rb") as audio_file:
                result = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                )
            os.unlink(tmp_path)
            return result.text

        except Exception as e:
            print(f"[AUDIO] API transcription error: {e}")
            return ""


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == "__main__":
    print("AudioListener Test")
    print("=" * 50)

    config = AudioConfig()
    listener = AudioListener(config)

    if not listener.available:
        print("No audio library available. Install: pip install sounddevice")
        print("Testing buffer logic only...")

        # Test buffer without mic
        listener._available = True  # Fake for testing
        chunk = np.random.randn(1600).astype(np.float32) * 0.1
        listener._process_chunk(chunk)

        audio = listener.get_recent_audio(0.1)
        if audio is not None:
            print(f"  Buffer works: got {audio.shape} tensor")
        else:
            print("  Buffer returned None (torch not available)")

        print("\n[OK] AudioListener logic works (no mic connected)")
    else:
        print(f"Audio available. Listening for 5 seconds...")
        listener.start()
        time.sleep(5)
        listener.stop()

        audio = listener.get_recent_audio(1.0)
        if audio is not None:
            print(f"  Got audio: {audio.shape}, RMS={audio.abs().mean():.4f}")

        text = listener.get_transcription()
        if text:
            print(f"  Transcription: \"{text}\"")
        else:
            print("  No speech detected")

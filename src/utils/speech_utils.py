import torch
from transformers import pipeline
from pydub import AudioSegment
from pydub.playback import play
import soundfile as sf
import io
import time
import threading

# Load TTS model
synthesiser = pipeline("text-to-speech", model="microsoft/speecht5_tts")
speaker_embedding = torch.zeros(1, 512)

# Feedback control variables
last_feedback_time = 0
feedback_cooldown = 5
feedback_lock = threading.Lock()
is_speaking = False

def play_audio(audio_data, sampling_rate):
    """Plays the generated speech audio from memory."""
    global is_speaking
    try:
        is_speaking = True
        audio = AudioSegment.from_file(io.BytesIO(audio_data), format="wav")
        play(audio)
    finally:
        is_speaking = False

def generate_and_play_speech(feedback_text):
    """Generates speech in parallel and plays it immediately."""
    try:
        # Generate AI speech
        speech = synthesiser(feedback_text, forward_params={"speaker_embeddings": speaker_embedding})
        print(f"Generated Speech: {feedback_text}")

        # Store audio in-memory instead of saving to disk
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, speech["audio"], samplerate=speech["sampling_rate"], format="wav")
        audio_buffer.seek(0)

        # Play audio in background thread
        audio_thread = threading.Thread(target=play_audio, args=(audio_buffer.read(), speech["sampling_rate"]))
        audio_thread.start()

    except Exception as e:
        print(f"Error generating speech: {e}")

def give_feedback(feedback_text):
    """Handles non-blocking speech playback with cooldown."""
    global last_feedback_time, is_speaking

    with feedback_lock:
        current_time = time.time()
        if current_time - last_feedback_time < feedback_cooldown or is_speaking:
            return 

        last_feedback_time = current_time
        speech_thread = threading.Thread(target=generate_and_play_speech, args=(feedback_text,))
        speech_thread.start()

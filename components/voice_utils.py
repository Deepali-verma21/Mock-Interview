import streamlit.components.v1 as components
import speech_recognition as sr
import pyttsx3
import os
import io
import tempfile

# Returned by transcribe_audio_bytes() when no real transcription could be
# produced (no client, and the offline fallback failed). Callers must check
# for this sentinel and NOT treat it as actual spoken content / submit it as
# an answer.
TRANSCRIPTION_UNAVAILABLE = "__TRANSCRIPTION_UNAVAILABLE__"

def detect_audio_mime_type(audio_bytes: bytes) -> str:
    """Detects audio container MIME type from raw byte magic headers."""
    if not audio_bytes:
        return "audio/wav"
    if audio_bytes.startswith(b"RIFF"):
        return "audio/wav"
    elif audio_bytes.startswith(b"\x1a\x45\xdf\xa3"):
        return "audio/webm"
    elif audio_bytes.startswith(b"OggS"):
        return "audio/ogg"
    elif audio_bytes.startswith(b"ID3") or audio_bytes.startswith(b"\xff\xfb"):
        return "audio/mp3"
    return "audio/webm"

def transcribe_audio_bytes(audio_bytes: bytes, client=None) -> str:
    """
    Transcribes audio bytes recorded via Streamlit's native st.audio_input.
    Dynamically identifies MIME type (audio/webm, audio/wav, audio/ogg).
    """
    if not audio_bytes:
        return ""

    mime_type = detect_audio_mime_type(audio_bytes)

    # 1. Gemini API Multimodal Audio Processing (Handles WebM, WAV, OGG natively)
    if client:
        try:
            from google.genai import types
            audio_part = types.Part.from_bytes(
                data=audio_bytes,
                mime_type=mime_type
            )
            prompt = (
                "You are an expert audio transcriber. Transcribe the user's spoken answer accurately word for word. "
                "Do not summarize or add commentary. Return ONLY the exact transcribed text."
            )
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[audio_part, prompt]
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            pass

    # 2. SpeechRecognition fallback (For WAV files)
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            if text:
                return text.strip()
    except Exception:
        pass

    # 3. No transcription available — return a sentinel, NOT a human-readable
    # note. Human-readable text here would previously get silently submitted
    # as if it were the candidate's actual spoken answer.
    return TRANSCRIPTION_UNAVAILABLE

def render_speech_to_text_widget():
    """Renders browser Web Speech API dictation box."""
    html_code = """
    <div style="font-family: Arial, sans-serif; background-color: #1e1e2e; color: #cdd6f4; padding: 14px; border-radius: 12px; margin-bottom: 12px; border: 1px solid #45475a;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <button id="micBtn" onclick="toggleDictation()" style="background-color: #89b4fa; color: #11111b; border: none; padding: 8px 18px; border-radius: 8px; font-weight: bold; cursor: pointer; display: flex; align-items: center; gap: 6px; font-size: 0.95em;">
                    <span id="micIcon">🎙️</span> <span id="micStatus">Start Dictation</span>
                </button>
                <span id="recordingText" style="font-size: 0.88em; color: #a6adc8; font-style: italic;">Click to speak into microphone...</span>
            </div>
            <div style="display: flex; gap: 12px; font-size: 0.85em; color: #bac2de;">
                <span>📝 Words: <strong id="wordCount" style="color:#89b4fa;">0</strong></span>
            </div>
        </div>
        <div id="transcriptBox" style="margin-top: 10px; font-size: 0.95em; line-height: 1.4; min-height: 48px; max-height: 120px; overflow-y: auto; padding: 10px; background: #181825; border-radius: 8px; color: #a6e3a1; border: 1px dashed #585b70;">
            Speech transcript will stream here as you speak...
        </div>
    </div>

    <script>
        var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        var recognition = null;
        var isRecording = false;

        if (SpeechRecognition) {
            recognition = new SpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = 'en-US';

            recognition.onstart = function() {
                isRecording = true;
                document.getElementById('micStatus').innerText = 'Listening... Click to Stop';
                document.getElementById('micBtn').style.backgroundColor = '#f38ba8';
                document.getElementById('recordingText').innerText = 'Recording active. Speak clearly.';
            };

            recognition.onend = function() {
                isRecording = false;
                document.getElementById('micStatus').innerText = 'Start Dictation';
                document.getElementById('micBtn').style.backgroundColor = '#89b4fa';
                document.getElementById('recordingText').innerText = 'Recording stopped.';
            };

            recognition.onresult = function(event) {
                var interimTranscript = '';
                var finalTranscript = '';
                for (var i = event.resultIndex; i < event.results.length; ++i) {
                    if (event.results[i].isFinal) {
                        finalTranscript += event.results[i][0].transcript;
                    } else {
                        interimTranscript += event.results[i][0].transcript;
                    }
                }
                var fullText = finalTranscript || interimTranscript;
                if (fullText) {
                    document.getElementById('transcriptBox').innerText = fullText;
                    var words = fullText.trim().split(/\s+/).filter(w => w.length > 0);
                    document.getElementById('wordCount').innerText = words.length;
                }
            };

            recognition.onerror = function(event) {
                document.getElementById('recordingText').innerText = 'Mic status: ' + event.error;
            };
        }

        function toggleDictation() {
            if (!recognition) {
                alert("Browser Speech Recognition not available in this context.");
                return;
            }
            if (isRecording) {
                recognition.stop();
            } else {
                recognition.start();
            }
        }
    </script>
    """
    return components.html(html_code, height=140)

def render_browser_tts(text_to_speak: str, button_label: str = "🔊 Read Out Loud"):
    import html
    safe_text = html.escape(text_to_speak).replace("\n", " ")
    
    html_code = f"""
    <div style="margin-top: 6px;">
        <button onclick="speakText()" style="background-color: #313244; color: #89b4fa; border: 1px solid #45475a; padding: 6px 14px; border-radius: 6px; font-size: 0.85em; font-weight: 600; cursor: pointer; display: inline-flex; align-items: center; gap: 6px;">
            <span>🔊</span> <span>{button_label}</span>
        </button>
    </div>
    <script>
        function speakText() {{
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
                var utterance = new SpeechSynthesisUtterance("{safe_text}");
                utterance.rate = 1.0;
                utterance.pitch = 1.0;
                utterance.lang = 'en-US';
                window.speechSynthesis.speak(utterance);
            }} else {{
                alert("Text-to-speech is not supported in your browser.");
            }}
        }}
    </script>
    """
    return components.html(html_code, height=42)

def speak_text(text: str):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        voices = engine.getProperty('voices')
        if voices:
            engine.setProperty('voice', voices[0].id)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as fp:
            temp_filename = fp.name
        
        engine.save_to_file(text, temp_filename)
        engine.runAndWait()
        
        with open(temp_filename, 'rb') as f:
            audio_bytes = f.read()
        
        os.remove(temp_filename)
        return audio_bytes
    except Exception:
        return None
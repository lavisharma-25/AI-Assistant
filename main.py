import os
import whisper
import pyttsx3
import pyaudio
import wave
from dotenv import load_dotenv
import google.generativeai as genai
load_dotenv()

# Initialize Whisper Model for Speech-to-Text
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")

# Initialize TTS engine (pyttsx3)
tts_engine = pyttsx3.init()

# Initialize the Gemini 1.5 Flash model (local, free)
print("Loading Gemini 1.5 Flash model for text generation...")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model_name = "gemini-1.5-flash"
# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}
model = genai.GenerativeModel(
    model_name=model_name,
    generation_config=generation_config,
    system_instruction=(
        """
        You are an intelligent assistant capable of understanding and responding to human speech. Your role is to process the user's spoken input, convert it to text, understand the question or request, and respond in a helpful and conversational manner. You should:
        - Understand a wide range of topics.
        - Provide accurate, relevant, and concise information.
        - Answer questions in a friendly and engaging manner.
        - Avoid long-winded explanations and keep answers clear and concise.
        - If the user asks for complex information, break it down into simple steps.
        - Always be polite and professional.
        - Respond as quickly as possible without unnecessary delays.
        
        When interacting, make sure to:
        - Acknowledge the user's input with positive reinforcement.
        - Avoid speaking in technical jargon unless specifically asked to.
        - Be capable of answering factual questions, providing suggestions, and helping with inquiries in various domains, including science, technology, arts, and general knowledge.
        """
    )
)

# Audio settings for pyaudio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper works best with 16 kHz audio
CHUNK = 1024


# Function to record live audio
def record_audio(seconds=5):
    print("Recording... Speak now!")
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    for _ in range(0, int(RATE / CHUNK * seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording complete!")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save recorded audio to a file
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    audio_file = os.path.join(current_script_path, "Audio_Data", "live_audio.wav")
    wf = wave.open(audio_file, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return audio_file


# Function to transcribe audio (speech-to-text)
def transcribe_audio(audio_file):
    print("Transcribing audio...")
    result = whisper_model.transcribe(audio_file)
    text = result["text"]
    print(f"Transcription: {text}")
    return text


# Function to generate model response using Gemini 1.5 Flash
def get_gpt_response(text_input):
    print("Processing input with Gemini 1.5 Flash...")
    response = model.generate_content(text_input)
    print(f"Response: {response.text}")
    return response.text


# Function to convert text to speech and play it
def text_to_speech(text):
    print(f"Speaking: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()


# Main Function
def main():
    while True:
        print("\n--- Starting Chatbot ---")

        # Step 1: Record audio and convert to text
        audio_file = record_audio(seconds=5)  # Adjust for longer or shorter input
        text_input = transcribe_audio(audio_file)

        # Step 2: Send the text input to Gemini 1.5 Flash for processing
        response = get_gpt_response(text_input)

        # Step 3: Convert the model's response to speech
        text_to_speech(response)

        # Option to exit the loop
        cont = input("Do you want to continue? (y/n): ").lower()
        if cont != 'y':
            break


if __name__ == "__main__":
    main()

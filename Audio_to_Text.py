from pydub import AudioSegment
import speech_recognition as sr

# # Load the audio file
# audio_path = 'audio.mp3'
# audio = AudioSegment.from_mp3(audio_path)
#
# # Convert to wav format as required by the speech_recognition library
# wav_path = audio_path.replace(".mp3", ".wav")
# audio.export(wav_path, format="wav")
#
# # Initialize the recognizer
# r = sr.Recognizer()
#
# # Recognize the audio
# with sr.AudioFile(wav_path) as source:
#     audio_data = r.record(source)
#     try:
#         # Attempt to convert audio to text
#         text = r.recognize_google(audio_data, language='en-US')  # Assuming the language is Chinese based on the file name
#     except sr.UnknownValueError:
#         # Error handling for unintelligible speech
#         text = "Audio was not clear enough to transcribe."
#     except sr.RequestError as e:
#         # Error handling for issues with the API
#         text = f"Could not request results from Google Speech Recognition service; {e}"
#
# text



# Initialize recognizer class (for recognizing the speech)
r = sr.Recognizer()

# Load your WAV file
with sr.AudioFile('D:\Project\pythonProject\\0306.wav') as source:
    # Listen for the data (load audio to memory)
    audio_data = r.record(source)
    # Recognize (convert from speech to text)
    text = r.recognize_google(audio_data)
    print(text)
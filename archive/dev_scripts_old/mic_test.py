# Old test version, superseded by main.py.
import speech_recognition as sr

r = sr.Recognizer()
r.energy_threshold = 100
r.dynamic_energy_threshold = False

print("Speak now...")
with sr.Microphone() as source:
    audio = r.listen(source, timeout=10)
    print("Got audio!")
    text = r.recognize_google(audio)
    print(f"You said: {text}")
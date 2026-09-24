import pyttsx3
import speech_recognition as sr
import threading
import queue
import time

class VoiceAssistant:

    def __init__(self):
        # ── Same settings that worked in quick_test ──
        self.is_listening = False
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold         = 100
        self.recognizer.dynamic_energy_threshold = False
        self.recognizer.pause_threshold          = 0.8

        # Cheap one-time check: does a microphone exist / is it free?
        # sr.Microphone() queries PyAudio for a device even before the
        # stream is opened, so a missing/busy mic raises here rather
        # than only failing later inside listen_once(). Lets the app
        # show "no microphone" up front instead of silently listening
        # into the void forever.
        self.mic_available = False
        self.mic_error      = None
        try:
            sr.Microphone()
            self.mic_available = True
        except Exception as e:
            self.mic_error = str(e)
            print(f"[voice] No usable microphone ({e}); voice control will stay disabled.")

        # ── State ──
        self.is_speaking   = False
        self.command_queue = queue.Queue()
        self.running       = True

        # ── Command keywords ──
        self.keyword_map = {
            "face_shape"   : [
                "face", "shape", "oval", "round",
                "square", "heart", "oblong", "jaw",
                "analyze", "detect"
            ],
            "skin_tone"    : [
                "skin", "tone", "color", "colour",
                "complexion", "dark", "light", "medium"
            ],
            "glasses"      : [
                "glass", "glasses", "specs", "frame",
                "lens", "classes", "spectacle",
                "eyewear", "eye", "try"
            ],
            "next_glasses" : [
                "next", "change", "switch",
                "another", "different"
            ],
            "prev_glasses" : [
                "previous", "prev", "back", "last"
            ],
            "outfit"       : [
                "wear", "outfit", "cloth",
                "dress", "wearing", "colors",
                "colours", "suggest colors"
            ],
            "hairstyle"    : [
                "hair", "style", "cut",
                "hairstyle", "haircut"
            ],
            "greeting"     : [
                "hello", "hi", "hey",
                "morning", "afternoon"
            ],
            "help"         : [
                "help", "assist", "guide",
                "command", "what can"
            ],
            "quit"         : [
                "quit", "exit", "bye",
                "goodbye", "stop", "close"
            ],
        }

        print("Voice Assistant ready!")

    # ──────────────────────────────────────
    # Speak
    # ──────────────────────────────────────
    def speak(self, text):
        """Speak and wait until done."""
        print(f"Assistant: {text}")
        self.is_speaking = True
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate',   150)
            engine.setProperty('volume', 1.0)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"TTS error: {e}")
        finally:
            self.is_speaking = False

    # ──────────────────────────────────────
    # Listen
    # ──────────────────────────────────────
    def listen_once(self):
        """Listen for one command and return text."""
        try:
            with sr.Microphone() as source:
                print("Listening...")
                audio = self.recognizer.listen(
                    source,
                    timeout=6,
                    phrase_time_limit=5
                )
                text = self.recognizer.recognize_google(
                    audio,
                    language='en-US'
                ).lower().strip()
                print(f"You said: {text}")
                return text

        except sr.WaitTimeoutError:
            print("No speech detected")
            return None
        except sr.UnknownValueError:
            print("Could not understand")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None

    # ──────────────────────────────────────
    # Match Command
    # ──────────────────────────────────────
    def match_command(self, text):
        """Match text to command using keywords."""
        if not text:
            return None

        words = text.lower().split()

        for command, keywords in self.keyword_map.items():
            for word in words:
                if word in keywords:
                    return command
            # Also check full text
            for keyword in keywords:
                if keyword in text:
                    return command

        return "unknown"

    # ──────────────────────────────────────
    # Response
    # ──────────────────────────────────────
    def get_response(self, command, context=None):
        """Get spoken response for command."""
        ctx        = context or {}
        face_shape = ctx.get('face_shape', None)
        skin_tone  = ctx.get('skin_tone',  None)
        gender     = ctx.get('gender',     'male')

        shape_advice = {
            "Oval"   : "Most hairstyles suit you. "
                       "Try layers or curtain bangs.",
            "Round"  : "Try longer styles and "
                       "rectangle glasses.",
            "Square" : "Soft waves and round glasses "
                       "suit you well.",
            "Heart"  : "Chin length styles and "
                       "aviator glasses are perfect.",
            "Oblong" : "Voluminous styles and wide "
                       "glasses frames suit you.",
        }

        tone_male = {
            "Very Light" : "Navy, forest green and "
                           "burgundy suit you best.",
            "Light"      : "Teal, olive and warm "
                           "browns look great.",
            "Medium"     : "Jewel tones, mustard "
                           "and white are perfect.",
            "Tan"        : "White, royal blue and "
                           "charcoal suit you well.",
            "Brown"      : "White, sky blue and "
                           "emerald look great.",
            "Deep"       : "White, red and gold "
                           "make you stand out.",
        }

        tone_female = {
            "Very Light" : "Dusty rose, burgundy "
                           "and navy suit you best.",
            "Light"      : "Coral, teal and warm "
                           "browns look gorgeous.",
            "Medium"     : "Jewel tones, mustard "
                           "and terracotta are stunning.",
            "Tan"        : "White, hot pink and "
                           "orange suit you well.",
            "Brown"      : "Fuchsia, yellow and "
                           "sky blue look amazing.",
            "Deep"       : "White, electric blue "
                           "and gold make you shine.",
        }

        tone_advice = tone_male \
            if gender == 'male' \
            else tone_female

        if command == "greeting":
            return ("Hello! I am your AI grooming "
                    "assistant. How can I help you?")

        elif command == "help":
            return ("You can ask me about your face "
                    "shape, skin tone, glasses, "
                    "hairstyle or outfit colors.")

        elif command == "face_shape":
            if face_shape:
                return (
                    f"Your face shape is {face_shape}. "
                    + shape_advice.get(face_shape, "")
                )
            return ("Please face the camera straight "
                    "so I can detect your face shape.")

        elif command == "skin_tone":
            if skin_tone:
                return (
                    f"Your skin tone is {skin_tone}. "
                    + tone_advice.get(skin_tone, "")
                )
            return ("Stay still so I can analyze "
                    "your skin tone.")

        elif command == "glasses":
            return ("Showing glasses for your face. "
                    "Say next to switch styles.")

        elif command == "next_glasses":
            return "Switching to next style."

        elif command == "prev_glasses":
            return "Going to previous style."

        elif command == "outfit":
            if skin_tone:
                return (
                    f"For {skin_tone} skin, "
                    + tone_advice.get(skin_tone, "")
                )
            return ("Let me detect your skin tone "
                    "first.")

        elif command == "hairstyle":
            if face_shape:
                return (
                    f"For {face_shape} face, "
                    + shape_advice.get(face_shape, "")
                )
            return ("Let me detect your face shape "
                    "first.")

        elif command == "quit":
            return "Goodbye! Have a great day."

        else:
            return ("Sorry, I did not understand. "
                    "Say help for commands.")

    # ──────────────────────────────────────
    # Background Loop
    # ──────────────────────────────────────
    def _listen_loop(self):
        """Runs in background thread."""
        while self.running:
            if not self.is_speaking:
                text    = self.listen_once()
                command = self.match_command(text)
                if command:
                    self.command_queue.put(
                        (command, text)
                    )
            time.sleep(0.1)

    def start_background(self):
        """Start listening in background."""
        t = threading.Thread(
            target=self._listen_loop,
            daemon=True
        )
        t.start()
        print("Listening in background...")

    def stop(self):
        self.running = False

    def get_command(self):
        """Get next command. Non-blocking."""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None


# ──────────────────────────────────────────
# Standalone Test
# ──────────────────────────────────────────
def main():
    print("=" * 40)
    print("Voice Assistant Test")
    print("=" * 40)
    print("Commands to try:")
    print("  face shape / skin tone / glasses")
    print("  hairstyle / what should i wear")
    print("  hello / help / quit")
    print("=" * 40)

    assistant = VoiceAssistant()

    # Test context
    context = {
        'face_shape' : 'Oval',
        'skin_tone'  : 'Medium',
        'gender'     : 'male',
    }

    assistant.speak(
        "Voice assistant ready. Say a command!"
    )

    while True:
        try:
            text = assistant.listen_once()
            if not text:
                continue

            command = assistant.match_command(text)
            print(f"Command: {command}")

            response = assistant.get_response(
                command, context
            )
            assistant.speak(response)

            if command == 'quit':
                break

        except KeyboardInterrupt:
            print("\nStopped.")
            break


if __name__ == "__main__":
    main()
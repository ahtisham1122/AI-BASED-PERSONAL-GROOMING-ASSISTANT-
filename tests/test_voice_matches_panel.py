"""
Checks the voice assistant says the same face shape / skin tone the panel
shows, has advice for every label the models can produce, and routes
common phrases to the right command. No microphone needed.
Run: python tests/test_voice_matches_panel.py
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # project root

from app.face_shape_model import CLASS_NAMES
from app.voice_assistant import VoiceAssistant

SKIN_TONES = ["Very Light", "Light", "Medium", "Tan", "Brown", "Deep"]  # every label classify_skin_tone returns
va = VoiceAssistant()


def test_face_shape_answer_matches_panel():
    for shape in CLASS_NAMES:
        reply = va.get_response("face_shape", {"face_shape": shape})
        assert f"Your face shape is {shape}. " in reply
        assert len(reply) > len(f"Your face shape is {shape}. "), f"no advice for {shape}"


def test_skin_tone_answer_matches_panel():
    for gender in ("male", "female"):
        for tone in SKIN_TONES:
            reply = va.get_response("skin_tone", {"skin_tone": tone, "gender": gender})
            assert f"Your skin tone is {tone}. " in reply
            assert len(reply) > len(f"Your skin tone is {tone}. "), f"no advice for {tone} ({gender})"


def test_commands_route():
    phrases = {
        "what is my face shape": "face_shape",
        "show me glasses": "glasses",
        "next": "next_glasses",
        "suggest a hairstyle": "hairstyle",
        "goodbye": "quit",
    }
    for text, expected in phrases.items():
        assert va.match_command(text) == expected, f"{text!r} -> {va.match_command(text)}"


if __name__ == '__main__':
    test_face_shape_answer_matches_panel()
    test_skin_tone_answer_matches_panel()
    test_commands_route()
    print("test_voice_matches_panel: all passed")

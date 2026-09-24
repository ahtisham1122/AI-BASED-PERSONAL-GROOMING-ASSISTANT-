import cv2
import numpy as np
import os
import threading
import time
import logging
import traceback

from voice_assistant import VoiceAssistant
from face_tracking import (
    create_landmarker, FrameTimestamper, detect_landmarks,
    get_head_yaw, StableDetector, OneEuroFilter,
)
from face_shape_model import (
    MODE as FACE_SHAPE_MODE, classify_face_shape, CNNFaceShapeClassifier,
    is_frontal_face, is_face_near_edge, save_debug_crop, get_last_inference_ms,
)
from skin_tone import correct_lighting, sample_skin_color, classify_skin_tone
from recommendations import get_suggested_glasses
from ar_overlay import load_glasses, overlay_glasses
from ui import (
    show_gender_selection, UIRenderer, build_ui_state, draw_notification, draw_rating_prompt,
    hit_test_tab, next_tab,
)
import feedback
from performance_logger import PerformanceLogger

# Any unexpected exception during the main loop gets logged here (with
# a full traceback) instead of taking the app down — see the try/except
# wrapped around the per-frame body in main().
logging.basicConfig(
    filename='app.log',
    level=logging.ERROR,
    format='%(asctime)s %(levelname)s %(message)s',
)


def save_snapshot(frame):
    """Save current frame as image."""
    folder = 'snapshots'
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(
        folder,
        f"snapshot_{int(time.time())}.jpg"
    )
    cv2.imwrite(filename, frame)
    print(f"Snapshot saved: {filename}")
    return filename


def main():
    # ── Gender selection ──
    gender = show_gender_selection()

    # ── Load glasses ──
    glasses_list, glasses_names = load_glasses('assets/glasses')
    glasses_idx = 0
    suggested   = list(range(len(glasses_list)))

    # ── Camera ──
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        camera_index = 1
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("ERROR: No camera!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # ── Voice assistant ──
    voice         = VoiceAssistant()
    voice_enabled = False
    voice_status  = "off"
    if not voice.mic_available:
        print("[voice] No microphone detected — voice control disabled for this session.")

    # ── Stable detectors / smoothing ──
    shape_det  = StableDetector(30, 22)  # only used in rule-based fallback mode
    tone_det   = StableDetector(40, 30)
    smoother   = OneEuroFilter()

    # ── Face shape: CNN on a background thread, or the rule-based fallback ──
    cnn_classifier = CNNFaceShapeClassifier() if FACE_SHAPE_MODE != 'rule-based' else None
    rule_based_mode = FACE_SHAPE_MODE == 'rule-based'

    # ── UI renderer (owns fonts + cached top/panel/bottom images) ──
    ui_renderer = UIRenderer()
    print(f"[ui] Rich PIL UI: {'on' if ui_renderer.pil_ok else 'off (falling back to plain cv2 UI)'}")

    # ── App state ──
    face_shape            = None
    face_shape_confidence = None
    analyzing         = FACE_SHAPE_MODE != 'rule-based'
    analysis_progress = (0, 1)
    near_edge       = False
    multiple_faces  = False
    face_present    = False
    skin_tone       = None
    skin_tone_low_confidence = False
    frame_count     = 0
    avg_bgr         = np.array([180.0, 150.0, 130.0])
    notification    = ""
    notif_timer     = 0
    landmarks       = None
    corrected       = None

    # ── Feedback rating flow (off/inactive until R is pressed) ──
    rating_stage              = None  # None, 'hairstyle', or 'glasses'
    rating_hair_suggestion    = None
    rating_glasses_suggestion = None

    # ── Right-panel tab (which card the bottom tab row is showing) ──
    active_tab      = 'face'
    last_voice_text = None

    # ── FPS tracking (smoothed) ──
    fps_ema   = 30.0
    last_tick = time.perf_counter()

    # ── Performance/stress-test logging (FYP report sections 6.8/6.9) ──
    perf_logger = PerformanceLogger()

    # ── Camera health (see stress-test case 11: unplugged/covered camera) ──
    camera_ok            = True
    consecutive_failures = 0

    print("App started!")
    print("N=Next glasses  P=Prev  V=Toggle voice  S=Save  A=Re-analyze  "
          "R=Rate suggestions  Shift+R=Clear feedback  C=Reset  D=Debug crop  "
          "L=Log performance  [ ]=Switch tab  Q=Quit")

    timestamper = FrameTimestamper()

    # ── Window + mouse click handling for the bottom tab row ──
    window_name = "AI Grooming Assistant"
    cv2.namedWindow(window_name)

    def on_click(event, x, y, flags, param):
        nonlocal active_tab
        if event == cv2.EVENT_LBUTTONDOWN:
            tab = hit_test_tab(x, y)
            if tab:
                active_tab = tab

    cv2.setMouseCallback(window_name, on_click)

    with create_landmarker() as landmarker:
        while True:
            ret, frame = cap.read()

            if not ret:
                consecutive_failures += 1
                if camera_ok:
                    print("[camera] Lost the camera feed, retrying...")
                    camera_ok = False
                # Don't reopen every single failed frame (that would
                # hammer the driver) — retry every ~30 failed reads.
                if consecutive_failures % 30 == 1:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = cv2.VideoCapture(camera_index)

                lost_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(lost_frame, "Camera lost - retrying...", (60, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(lost_frame, "Press Q to quit", (60, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                cv2.imshow("AI Grooming Assistant", lost_frame)
                key = cv2.waitKey(200) & 0xFF
                if key == ord('q'):
                    break
                continue

            if not camera_ok:
                print("[camera] Camera feed recovered")
            camera_ok            = True
            consecutive_failures = 0

            frame      = cv2.flip(frame, 1)
            h, w, _    = frame.shape
            frame_count += 1

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - last_tick, 1e-6)
            fps_ema = 0.9 * fps_ema + 0.1 * instant_fps
            last_tick = now
            perf_logger.tick(fps_ema, analyzing, get_last_inference_ms())

            # Everything below touches the model pipeline, image
            # processing, and key handling — any of it failing
            # unexpectedly gets logged to app.log with a traceback and
            # the loop just moves on to the next frame, rather than
            # taking the whole app down.
            try:
                corrected      = correct_lighting(frame)
                rgb            = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
                landmarks, multiple_faces = detect_landmarks(landmarker, timestamper, rgb)
                face_present   = landmarks is not None

                if landmarks is not None:
                    # ── Rule-based face shape + skin tone, every 15 frames ──
                    if frame_count % 15 == 0:
                        yaw = get_head_yaw(landmarks)
                        if yaw < 0.08:
                            if FACE_SHAPE_MODE == 'rule-based':
                                raw = classify_face_shape(landmarks, w, h)
                                face_shape = shape_det.update(raw)

                            sampled, low_conf = sample_skin_color(corrected, landmarks, w, h)
                            if sampled is not None:
                                avg_bgr        = sampled
                                raw_tone       = classify_skin_tone(avg_bgr)
                                skin_tone      = tone_det.update(raw_tone)
                                skin_tone_low_confidence = low_conf

                    # ── CNN face shape: hand off a frontal frame every ~5
                    # frames (never wait on it), collect toward the
                    # analysis lock, and read back whatever the background
                    # thread currently has. `corrected` is the clean
                    # lighting-corrected frame — glasses/UI are drawn onto
                    # `frame` separately, so the CNN never sees them. ──
                    if cnn_classifier is not None:
                        cnn_classifier.note_face_seen(True)
                        near_edge = is_face_near_edge(landmarks, w, h)
                        if (frame_count % 5 == 0 and not near_edge
                                and is_frontal_face(landmarks, w, h)):
                            cnn_classifier.submit(corrected, landmarks, w, h)

                        label, confidence, locked, collected, total = cnn_classifier.get_result()
                        analyzing         = not locked
                        analysis_progress = (collected, total)
                        if locked:
                            face_shape            = label
                            face_shape_confidence = confidence

                    if face_shape:
                        suggested = get_suggested_glasses(face_shape, len(glasses_list))

                    # ── AR Glasses ──
                    if glasses_list:
                        lo = landmarks[33]
                        ro = landmarks[263]
                        lc = landmarks[468] \
                            if len(landmarks) > 468 \
                            else landmarks[33]
                        rc = landmarks[473] \
                            if len(landmarks) > 468 \
                            else landmarks[263]

                        lc_px = (int(lc.x * w), int(lc.y * h))
                        rc_px = (int(rc.x * w), int(rc.y * h))
                        lo_px = (int(lo.x * w), int(lo.y * h))
                        ro_px = (int(ro.x * w), int(ro.y * h))

                        eye_dist = abs(ro_px[0] - lo_px[0])
                        gw       = int(eye_dist * 1.8)

                        if gw > 10:
                            gi       = glasses_list[glasses_idx]
                            oh, ow   = gi.shape[:2]
                            aspect   = oh / ow
                            gh       = int(gw * aspect)
                            cx       = (lc_px[0] + rc_px[0]) // 2
                            cy       = (lc_px[1] + rc_px[1]) // 2
                            angle    = np.degrees(
                                np.arctan2(
                                    rc_px[1] - lc_px[1],
                                    rc_px[0] - lc_px[0]
                                )
                            )
                            x1 = cx - gw // 2
                            y1 = cy - int(gh * 0.45)

                            vals     = np.array(
                                [x1, y1, gw, gh, angle],
                                dtype=float
                            )
                            smoothed = smoother.smooth(vals)

                            frame = overlay_glasses(
                                frame,
                                gi,
                                int(smoothed[0]),
                                int(smoothed[1]),
                                int(smoothed[2]),
                                int(smoothed[3]),
                                float(smoothed[4])
                            )
                else:
                    smoother.reset()
                    near_edge = False
                    if cnn_classifier is not None:
                        cnn_classifier.note_face_seen(False)

                # ── Voice commands ──
                if voice_enabled:
                    voice_status = "listening" \
                        if voice.is_listening \
                        else "on"
                    cmd = voice.get_command()
                    if cmd:
                        command, text = cmd
                        last_voice_text = text
                        context = {
                            'face_shape' : face_shape,
                            'skin_tone'  : skin_tone,
                            'gender'     : gender,
                        }
                        response = voice.get_response(
                            command, context
                        )

                        # Handle action commands
                        if command == 'next_glasses' \
                                and glasses_list:
                            glasses_idx = (
                                glasses_idx + 1
                            ) % len(glasses_list)
                            smoother.reset()
                        elif command == 'prev_glasses' \
                                and glasses_list:
                            glasses_idx = (
                                glasses_idx - 1
                            ) % len(glasses_list)
                            smoother.reset()
                        elif command == 'quit':
                            voice.speak(response)
                            break

                        # Speak in background
                        t = threading.Thread(
                            target=voice.speak,
                            args=(response,),
                            daemon=True
                        )
                        t.start()

                        notification = f"Voice: {text}"
                        notif_timer  = 60

                # ── Draw UI ──
                cnn_busy = cnn_classifier.is_busy() if cnn_classifier is not None else False
                skin_swatch_rgb = (int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0])) if skin_tone else None
                ui_state = build_ui_state(
                    gender, voice_status, fps_ema,
                    face_shape, face_shape_confidence,
                    analyzing, analysis_progress, near_edge, cnn_busy,
                    skin_tone, skin_swatch_rgb,
                    glasses_idx, glasses_names, suggested, len(glasses_list),
                    face_present=face_present, multiple_faces=multiple_faces,
                    rule_based_mode=rule_based_mode,
                    skin_tone_low_confidence=skin_tone_low_confidence,
                    active_tab=active_tab, mic_available=voice.mic_available,
                    last_voice_text=last_voice_text,
                )
                frame = ui_renderer.render(frame, ui_state)
                display_h = frame.shape[0]

                # ── Notification ──
                if notif_timer > 0:
                    frame = draw_notification(frame, notification, display_h)
                    notif_timer -= 1

                # ── Rating prompt overlay ──
                if rating_stage is not None:
                    suggestion_name = (
                        rating_hair_suggestion if rating_stage == 'hairstyle'
                        else rating_glasses_suggestion
                    )
                    frame = draw_rating_prompt(frame, rating_stage, suggestion_name or "?")

                cv2.imshow(
                    "AI Grooming Assistant", frame
                )

                # ── Keys ──
                key = cv2.waitKey(1) & 0xFF

                if rating_stage is not None:
                    # Rating mode takes over the keyboard entirely (only
                    # digits 1-5 and Esc matter) so a stray keypress mid-
                    # rating can't also trigger something else, like
                    # switching glasses. Never allowed to crash the loop —
                    # any failure here just cancels the rating.
                    try:
                        if key == 27:  # Esc: skip this one
                            if rating_stage == 'hairstyle':
                                rating_stage = 'glasses'
                                rating_glasses_suggestion = (
                                    ui_state['glasses_types'][0]['name']
                                    if ui_state['glasses_types'] else None
                                )
                                if rating_glasses_suggestion is None:
                                    rating_stage = None
                            else:
                                rating_stage = None
                                notification = "Feedback skipped"
                                notif_timer  = 60
                        elif key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
                            rating_value = key - ord('0')
                            if rating_stage == 'hairstyle' and rating_hair_suggestion:
                                feedback.add_rating(face_shape, gender, skin_tone, 'hairstyle',
                                                     rating_hair_suggestion, rating_value)
                                rating_stage = 'glasses'
                                rating_glasses_suggestion = (
                                    ui_state['glasses_types'][0]['name']
                                    if ui_state['glasses_types'] else None
                                )
                                if rating_glasses_suggestion is None:
                                    rating_stage = None
                                    notification = "Thanks for your feedback!"
                                    notif_timer  = 60
                            elif rating_stage == 'glasses' and rating_glasses_suggestion:
                                feedback.add_rating(face_shape, gender, skin_tone, 'glasses',
                                                     rating_glasses_suggestion, rating_value)
                                rating_stage = None
                                notification = "Thanks for your feedback!"
                                notif_timer  = 60
                    except Exception as e:
                        print(f"[feedback] Rating flow error: {e}")
                        rating_stage = None

                elif key == ord('q'):
                    break

                elif key == ord('n') and glasses_list:
                    glasses_idx = (
                        glasses_idx + 1
                    ) % len(glasses_list)
                    smoother.reset()

                elif key == ord('p') and glasses_list:
                    glasses_idx = (
                        glasses_idx - 1
                    ) % len(glasses_list)
                    smoother.reset()

                elif key == ord('v'):
                    if not voice.mic_available:
                        notification = "Voice unavailable: no microphone detected"
                        notif_timer  = 90
                    else:
                        voice_enabled = not voice_enabled
                        if voice_enabled:
                            voice.start_background()
                            voice_status = "on"
                            notification = "Voice ON! Speak a command"
                            notif_timer  = 60
                            print("Voice enabled!")
                        else:
                            voice.stop()
                            voice_status = "off"
                            notification = "Voice OFF"
                            notif_timer  = 60
                            print("Voice disabled!")

                elif key == ord('s'):
                    filename = save_snapshot(frame)
                    notification = f"Saved: {filename}"
                    notif_timer  = 90

                elif key == ord('a'):
                    if cnn_classifier is not None:
                        cnn_classifier.restart()
                        notification = "Re-analyzing face shape..."
                        notif_timer  = 60

                elif key == ord('c'):
                    # Full manual reset: new person sitting down shouldn't
                    # have to wait for the CNN's own auto-reset timeout,
                    # and this is the only way to reset skin tone / the
                    # rule-based fallback, neither of which auto-resets.
                    shape_det = StableDetector(30, 22)
                    tone_det  = StableDetector(40, 30)
                    skin_tone = None
                    skin_tone_low_confidence = False
                    if cnn_classifier is not None:
                        cnn_classifier.restart()
                    notification = "Reset — analyzing from scratch"
                    notif_timer  = 60

                elif key == ord('r'):
                    # Rate the current top hairstyle + glasses suggestions
                    # (only makes sense once there's an actual locked
                    # result to rate).
                    try:
                        if not analyzing and face_shape and ui_state['hair_recs']:
                            rating_stage           = 'hairstyle'
                            rating_hair_suggestion = ui_state['hair_recs'][0]['name']
                        else:
                            notification = "Lock a face shape result first"
                            notif_timer  = 60
                    except Exception as e:
                        print(f"[feedback] Couldn't start rating flow: {e}")
                        rating_stage = None

                elif key == ord('R'):
                    # Shift+R: wipe feedback.json for a clean demo.
                    try:
                        feedback.reset()
                        notification = "Feedback cleared"
                        notif_timer  = 60
                    except Exception as e:
                        print(f"[feedback] Couldn't clear feedback: {e}")

                elif key == ord('d'):
                    if landmarks is not None:
                        path = save_debug_crop(corrected, landmarks, w, h)
                        notification = (
                            f"Debug crop saved: {path}" if path
                            else "Face not frontal/large enough for a debug crop"
                        )
                    else:
                        notification = "No face detected"
                    notif_timer = 90

                elif key == ord('l'):
                    perf_logger.save_log()
                    notification = "Performance results saved"
                    notif_timer  = 90

                elif key == ord('['):
                    active_tab = next_tab(active_tab, -1)

                elif key == ord(']'):
                    active_tab = next_tab(active_tab, 1)

            except Exception:
                logging.error("Unhandled exception in main loop:\n%s", traceback.format_exc())
                print("[ERROR] Unexpected error this frame (see app.log) - continuing")
                continue

    cap.release()
    cv2.destroyAllWindows()
    voice.stop()
    if cnn_classifier is not None:
        cnn_classifier.stop()
    print("App closed!")


if __name__ == "__main__":
    main()

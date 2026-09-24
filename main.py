import cv2
import numpy as np
import os
import threading
import time
import logging
import traceback

# Windows stretches/blurs the window when it thinks the app is DPI-
# unaware on a scaled display (125%/150% scaling is common on laptops),
# which is a big part of why the UI can look jagged. Telling Windows
# this process handles its own DPI (PROCESS_PER_MONITOR_DPI_AWARE = 2)
# before any window is created fixes that. try/except because
# shcore/SetProcessDpiAwareness only exists on Windows 8.1+ -- this
# must silently no-op on any other OS.
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

from app.voice_assistant import VoiceAssistant
from app.face_tracking import (
    create_landmarker, FrameTimestamper, detect_landmarks,
    get_head_yaw, StableDetector, OneEuroFilter,
    solve_head_pose, euler_to_rotation_matrix,
)
from app.face_shape_model import (
    MODE as FACE_SHAPE_MODE, classify_face_shape, CNNFaceShapeClassifier,
    is_frontal_face, is_face_near_edge, save_debug_crop, get_last_inference_ms,
)
from app.skin_tone import correct_lighting, sample_skin_color, classify_skin_tone
from app.recommendations import get_suggested_glasses
from app.ar_overlay import (
    load_glasses, compute_glasses_geometry, build_glasses_corners, warp_and_blend_glasses,
)
from app.ui import (
    show_gender_selection, UIRenderer, build_ui_state, draw_notification, draw_rating_prompt,
    hit_test_tab, next_tab,
)
from app import feedback
from app.paths import ASSETS_DIR, DATA_DIR
from app.performance_logger import PerformanceLogger

# Any unexpected exception during the main loop gets logged here (with
# a full traceback) instead of taking the app down — see the try/except
# wrapped around the per-frame body in main().
logging.basicConfig(
    filename=str(DATA_DIR / 'app.log'),
    level=logging.ERROR,
    format='%(asctime)s %(levelname)s %(message)s',
)


def save_snapshot(frame):
    """Save current frame as image."""
    folder = str(DATA_DIR / 'snapshots')
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
    # Loads from processed/ -- the prepare_glasses.py output (backgrounds
    # removed, lenses made see-through). Run `python tools/prepare_glasses.py`
    # by hand whenever new images are added to assets/glasses/.
    glasses_list, glasses_names = load_glasses(str(ASSETS_DIR / 'glasses' / 'processed'))
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

    # ── AR glasses smoothing: position/scale and head-pose angles get
    # separate One Euro filters since they behave differently (a small
    # position wobble matters more than a small angle wobble) — tuned
    # per the FYP spec's given constants. ──
    glasses_pos_filter   = OneEuroFilter(min_cutoff=1.0, beta=0.08)  # [center_x, center_y, width], pixels
    glasses_angle_filter = OneEuroFilter(min_cutoff=0.8, beta=0.05)  # [pitch, yaw, roll], degrees

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

    # ── AR glasses: off by default, toggled by the Glasses tab / G key ──
    glasses_on = False
    # Fully faded in (1.0) or out (0.0) -- ramps at 1/GLASSES_FADE_SECONDS
    # per second, so turning glasses on/off (or losing/regaining the
    # face) doesn't pop.
    glasses_alpha = 0.0
    GLASSES_FADE_SECONDS = 0.2
    # Past this yaw, the glasses would sit flat on a face we're mostly
    # seeing edge-on and look obviously wrong -- fade out instead.
    GLASSES_MAX_YAW_DEG = 35
    # Last successfully-placed corners, kept so a fade-out (face lost,
    # or turned away) has something to keep drawing at reduced alpha
    # instead of vanishing mid-fade.
    glasses_last_src = None
    glasses_last_dst = None
    glasses_last_asset = None

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
    print("N=Next glasses  P=Prev  G=Toggle glasses  V=Toggle voice  S=Save  A=Re-analyze  "
          "R=Rate suggestions  Shift+R=Clear feedback  C=Reset  D=Debug crop  "
          "L=Log performance  [ ]=Switch tab  Q=Quit")

    timestamper = FrameTimestamper()

    # ── Window + mouse click handling for the bottom tab row ──
    window_name = "AI Grooming Assistant"
    cv2.namedWindow(window_name)

    def on_click(event, x, y, flags, param):
        nonlocal active_tab, glasses_on
        if event == cv2.EVENT_LBUTTONDOWN:
            tab = hit_test_tab(x, y)
            if tab:
                active_tab = tab
                if tab == 'glasses':
                    # Clicking the Glasses tab both shows that panel
                    # AND toggles the AR overlay on/off (click again to
                    # turn it back off) -- same toggle the G key does.
                    glasses_on = not glasses_on

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
            frame_dt = max(now - last_tick, 1e-6)  # real elapsed time this frame took -- used below for the glasses fade and One Euro filters
            instant_fps = 1.0 / frame_dt
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

                # Head pose (yaw/pitch/roll), computed every single
                # frame a face is present -- never throttled/skipped
                # like the CNN or rule-based sampling below, since the
                # AR glasses need it every frame to stay jitter-free
                # and to know instantly when the head turns away.
                pose_ok, _rvec, _tvec, pose_angles = (False, None, None, None)
                if landmarks is not None:
                    pose_ok, _rvec, _tvec, pose_angles = solve_head_pose(landmarks, w, h)

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
                else:
                    near_edge = False
                    if cnn_classifier is not None:
                        cnn_classifier.note_face_seen(False)

                # ── AR Glasses: off by default, toggled on by the
                # Glasses tab / G key. Runs independently of the
                # landmarks/CNN block above because it needs to keep
                # animating (fading out) even on the very frame the
                # face disappears or turns too far away, rather than
                # just cutting off. ──
                yaw_deg = pose_angles[1] if pose_ok else 0.0
                glasses_should_show = (
                    glasses_on and bool(glasses_list) and landmarks is not None
                    and pose_ok and abs(yaw_deg) <= GLASSES_MAX_YAW_DEG
                )
                target_alpha = 1.0 if glasses_should_show else 0.0
                alpha_step = frame_dt / GLASSES_FADE_SECONDS
                if glasses_alpha < target_alpha:
                    glasses_alpha = min(target_alpha, glasses_alpha + alpha_step)
                else:
                    glasses_alpha = max(target_alpha, glasses_alpha - alpha_step)

                if glasses_alpha > 0.001 and glasses_list:
                    if landmarks is not None and pose_ok:
                        gw_raw, cx_raw, cy_raw = compute_glasses_geometry(landmarks, w, h)
                        pos_smoothed = glasses_pos_filter.smooth(
                            np.array([cx_raw, cy_raw, gw_raw], dtype=float), t=now)
                        angle_smoothed = glasses_angle_filter.smooth(
                            np.array(pose_angles, dtype=float), t=now)
                        R_smoothed = euler_to_rotation_matrix(*np.radians(angle_smoothed))
                        glasses_asset = glasses_list[glasses_idx]
                        glasses_last_src, glasses_last_dst = build_glasses_corners(
                            glasses_asset, pos_smoothed[2],
                            (pos_smoothed[0], pos_smoothed[1]), R_smoothed,
                        )
                        glasses_last_asset = glasses_asset
                        frame = warp_and_blend_glasses(
                            frame, glasses_asset, glasses_last_src, glasses_last_dst, glasses_alpha)
                    elif glasses_last_dst is not None:
                        # Face/pose unavailable this frame -- keep
                        # drawing at the last known placement while
                        # fading out, instead of vanishing mid-fade.
                        frame = warp_and_blend_glasses(
                            frame, glasses_last_asset, glasses_last_src, glasses_last_dst, glasses_alpha)
                elif glasses_last_dst is not None:
                    # Fully faded out -- clear the smoothing state so a
                    # face reappearing later starts fresh instead of
                    # interpolating in from a stale position.
                    glasses_pos_filter.reset()
                    glasses_angle_filter.reset()
                    glasses_last_src = glasses_last_dst = glasses_last_asset = None

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

                        # Handle action commands (glasses only actually
                        # change while they're turned on -- see the G
                        # key / Glasses tab toggle)
                        if command == 'next_glasses' \
                                and glasses_list and glasses_on:
                            glasses_idx = (
                                glasses_idx + 1
                            ) % len(glasses_list)
                        elif command == 'prev_glasses' \
                                and glasses_list and glasses_on:
                            glasses_idx = (
                                glasses_idx - 1
                            ) % len(glasses_list)
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

                elif key == ord('n') and glasses_list and glasses_on:
                    glasses_idx = (
                        glasses_idx + 1
                    ) % len(glasses_list)

                elif key == ord('p') and glasses_list and glasses_on:
                    glasses_idx = (
                        glasses_idx - 1
                    ) % len(glasses_list)

                elif key in (ord('g'), ord('G')):
                    glasses_on = not glasses_on

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

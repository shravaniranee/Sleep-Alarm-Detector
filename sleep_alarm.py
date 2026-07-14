import cv2
import mediapipe as mp
import numpy as np
import time
import os
import threading
from collections import deque

# ---------------------------------------------------------------------------
# Optional audio (pygame) — degrade gracefully if unavailable (no lib / no
# audio device, e.g. headless machines).
# ---------------------------------------------------------------------------
try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except Exception as e:  # ImportError OR pygame.error (no audio device)
    PYGAME_AVAILABLE = False
    print(f"[WARNING] pygame audio unavailable ({e}). Falling back to console beep.")

# ---------------------------------------------------------------------------
# Optional voice alerts (pyttsx3)
# ---------------------------------------------------------------------------
try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False
    print("[WARNING] pyttsx3 not found. Voice alerts disabled. Install with: pip install pyttsx3")

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------
EAR_THRESHOLD = 0.22          # below this the eyes are considered closed
EYE_CLOSED_SECONDS = 2.5      # how long eyes must stay closed to alarm

MAR_THRESHOLD = 0.60          # above this the mouth is considered wide open
YAWN_MIN_SECONDS = 0.6        # mouth must stay open this long to count as a yawn

HEAD_DROOP_DEG = 18.0         # |pitch| beyond this = head nodding forward/back
HEAD_DROOP_SECONDS = 2.0      # how long the head must droop to alarm

SCORE_ALARM_THRESHOLD = 85    # composite drowsiness score that triggers alarm
BLINK_MAX_SECONDS = 0.4       # closures shorter than this count as blinks
METRIC_WINDOW_SECONDS = 60.0  # rolling window for blink-rate / yawn-rate

SNOOZE_SECONDS = 20.0         # how long an alarm stays muted after a snooze
VOICE_REPEAT_SECONDS = 4.0    # gap between spoken "wake up" alerts
ALARM_RAMP_SECONDS = 5.0      # time for alarm volume to ramp 0.4 -> 1.0

ALARM_SOUND_FILE = r"dragon-studio-censor-beep-3-372460.mp3"

# ---------------------------------------------------------------------------
# MediaPipe FaceMesh landmark indices
# ---------------------------------------------------------------------------
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Inner-lip vertical pairs + mouth-corner horizontal, for MAR
MOUTH_VERTICAL_PAIRS = [(13, 14), (82, 87), (312, 317)]
MOUTH_CORNERS = (61, 291)

# Landmarks used for 6-point head-pose estimation
POSE_LANDMARKS = {
    "nose": 1,
    "chin": 152,
    "left_eye": 33,
    "right_eye": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}
# Generic 3D face model (mm), matching POSE_LANDMARKS order below
MODEL_POINTS_3D = np.array([
    (0.0,    0.0,    0.0),      # nose tip
    (0.0,   -330.0, -65.0),     # chin
    (-225.0, 170.0, -135.0),    # left eye corner
    (225.0,  170.0, -135.0),    # right eye corner
    (-150.0, -150.0, -125.0),   # left mouth corner
    (150.0,  -150.0, -125.0),   # right mouth corner
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def eye_aspect_ratio(landmarks, eye_indices, frame_w, frame_h):
    pts = [np.array([landmarks[idx].x * frame_w, landmarks[idx].y * frame_h])
           for idx in eye_indices]
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(landmarks, frame_w, frame_h):
    def pt(idx):
        return np.array([landmarks[idx].x * frame_w, landmarks[idx].y * frame_h])

    vertical = sum(np.linalg.norm(pt(a) - pt(b)) for a, b in MOUTH_VERTICAL_PAIRS)
    horizontal = np.linalg.norm(pt(MOUTH_CORNERS[0]) - pt(MOUTH_CORNERS[1]))
    if horizontal == 0:
        return 0.0
    return vertical / (2.0 * horizontal)


def head_pose_angles(landmarks, frame_w, frame_h):
    """Return (pitch, yaw, roll) in degrees, or None if it can't be solved.

    Pitch sign convention can vary by camera; we only use |pitch| for the
    droop heuristic and yaw *changes* for the head-shake gesture, so absolute
    sign does not matter.
    """
    image_points = np.array([
        (landmarks[POSE_LANDMARKS["nose"]].x * frame_w,  landmarks[POSE_LANDMARKS["nose"]].y * frame_h),
        (landmarks[POSE_LANDMARKS["chin"]].x * frame_w,  landmarks[POSE_LANDMARKS["chin"]].y * frame_h),
        (landmarks[POSE_LANDMARKS["left_eye"]].x * frame_w,  landmarks[POSE_LANDMARKS["left_eye"]].y * frame_h),
        (landmarks[POSE_LANDMARKS["right_eye"]].x * frame_w, landmarks[POSE_LANDMARKS["right_eye"]].y * frame_h),
        (landmarks[POSE_LANDMARKS["left_mouth"]].x * frame_w,  landmarks[POSE_LANDMARKS["left_mouth"]].y * frame_h),
        (landmarks[POSE_LANDMARKS["right_mouth"]].x * frame_w, landmarks[POSE_LANDMARKS["right_mouth"]].y * frame_h),
    ], dtype=np.float64)

    focal = frame_w
    center = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([
        [focal, 0,     center[0]],
        [0,     focal, center[1]],
        [0,     0,     1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    ok, rvec, _ = cv2.solvePnP(MODEL_POINTS_3D, image_points, camera_matrix,
                               dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = angles[0], angles[1], angles[2]
    # Normalise pitch to a sensible [-90, 90] range
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180
    return pitch, yaw, roll


# ---------------------------------------------------------------------------
# Audio / voice
# ---------------------------------------------------------------------------
def play_alarm(volume=0.4):
    if not PYGAME_AVAILABLE:
        print("\a[ALARM] Eyes closed too long! (No sound - pygame not installed)")
        return
    sound_path = os.path.join(os.path.dirname(__file__), ALARM_SOUND_FILE)
    if not os.path.isfile(sound_path):
        print(f"[ALARM] Sound file '{ALARM_SOUND_FILE}' not found - playing beep only.")
        print("\a")
        return
    try:
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.set_volume(volume)
        pygame.mixer.music.play(-1)
    except Exception as e:
        print(f"[ALARM] Could not play sound: {e}")


def set_alarm_volume(volume):
    if not PYGAME_AVAILABLE:
        return
    try:
        pygame.mixer.music.set_volume(max(0.0, min(1.0, volume)))
    except Exception:
        pass


def stop_alarm():
    if not PYGAME_AVAILABLE:
        return
    try:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
    except Exception:
        pass


_tts_lock = threading.Lock()


def speak(text):
    """Speak `text` asynchronously; silently no-op if TTS is busy/unavailable."""
    if not TTS_AVAILABLE:
        return

    def _run():
        if not _tts_lock.acquire(blocking=False):
            return  # already speaking, drop this one
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception:
            pass
        finally:
            _tts_lock.release()

    threading.Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# Drowsiness scoring
# ---------------------------------------------------------------------------
def compute_drowsiness_score(eye_elapsed, yawn_rate, droop_elapsed, blink_rate):
    """Blend the individual signals into a single 0-100 drowsiness score."""
    eye_c = min(eye_elapsed / EYE_CLOSED_SECONDS, 1.0) * 45.0
    yawn_c = min(yawn_rate / 3.0, 1.0) * 20.0           # 3 yawns/min -> maxed
    droop_c = min(droop_elapsed / HEAD_DROOP_SECONDS, 1.0) * 20.0
    # Both abnormally high blink rates and near-zero (staring) hint at fatigue
    if blink_rate > 25:
        blink_c = 15.0
    elif blink_rate > 18:
        blink_c = 8.0
    else:
        blink_c = 0.0
    return min(eye_c + yawn_c + droop_c + blink_c, 100.0)


def score_color(score):
    if score >= 80:
        return (0, 0, 220)      # red
    if score >= 50:
        return (0, 165, 255)    # orange
    return (0, 200, 50)         # green


# ---------------------------------------------------------------------------
# UI drawing
# ---------------------------------------------------------------------------
def draw_status_overlay(frame, alarm_active, stats):
    h, w = frame.shape[:2]
    score = stats["score"]

    if alarm_active:
        color, bg_color, label = (0, 0, 220), (0, 0, 180), "SLEEPING! WAKE UP!"
    elif stats["snoozed"]:
        color, bg_color, label = (0, 165, 255), (0, 120, 200), "SNOOZED"
    else:
        color, bg_color, label = (0, 200, 50), (0, 150, 40), "AWAKE"

    if alarm_active:
        pulse = int(abs(np.sin(time.time() * 4)) * 80)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.20 + pulse / 1000, frame, 0.80 - pulse / 1000, 0, frame)

    # ---- top status pill ----
    pill_w, pill_h = 360, 54
    pill_x = (w - pill_w) // 2
    pill_y = 14
    cv2.rectangle(frame, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), bg_color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), color, 2, cv2.LINE_AA)
    cv2.putText(frame, label, (pill_x + 20, pill_y + 36), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    dot_x, dot_y = pill_x - 30, pill_y + pill_h // 2
    cv2.circle(frame, (dot_x, dot_y), 14, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (dot_x, dot_y), 14, (255, 255, 255), 2, cv2.LINE_AA)

    # ---- drowsiness score gauge (top-left) ----
    gx, gy, gw, gh = 14, 14, 220, 16
    cv2.putText(frame, "Drowsiness", (gx, gy - 2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (gx, gy + 16), (gx + gw, gy + 16 + gh), (60, 60, 60), -1)
    fill = int(gw * score / 100.0)
    cv2.rectangle(frame, (gx, gy + 16), (gx + fill, gy + 16 + gh), score_color(score), -1)
    cv2.putText(frame, f"{int(score)}", (gx + gw + 8, gy + 16 + gh),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, score_color(score), 2, cv2.LINE_AA)

    # ---- bottom info bar ----
    bar_h = 74
    bar_y = h - bar_h
    cv2.rectangle(frame, (0, bar_y), (w, h), (20, 20, 20), -1)
    line1 = (f"EAR {stats['ear']:.3f} (<{EAR_THRESHOLD})   "
             f"MAR {stats['mar']:.2f} (>{MAR_THRESHOLD})   "
             f"Pitch {stats['pitch']:+.0f}deg")
    line2 = (f"Eyes closed {stats['eye_elapsed']:.1f}/{EYE_CLOSED_SECONDS}s   "
             f"Blinks {stats['blink_rate']:.0f}/min   "
             f"Yawns {stats['yawn_count']}")
    cv2.putText(frame, line1, (14, bar_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, line2, (14, bar_y + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    hint = "[Q] quit   [S] snooze   (or shake your head to snooze)"
    cv2.putText(frame, hint, (14, bar_y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

    if stats["snoozed"]:
        msg = f"Snoozed ({stats['snooze_left']:.0f}s)"
        cv2.putText(frame, msg, (w - 220, bar_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)

    # closed-eye progress line just above the bar
    if stats["eye_elapsed"] > 0 and not alarm_active:
        progress = min(stats["eye_elapsed"] / EYE_CLOSED_SECONDS, 1.0)
        bar_fill_w = int((w - 28) * progress)
        cv2.rectangle(frame, (14, bar_y - 4), (14 + bar_fill_w, bar_y - 1), (0, 165, 255), -1, cv2.LINE_AA)
    return frame


def draw_face_box(frame, landmarks, color, w, h):
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]
    pad = 10
    x1, y1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
    x2, y2 = min(w, max(xs) + pad), min(h, max(ys) + pad)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    corner, thick = 18, 3
    for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(frame, (cx, cy), (cx + dx * corner, cy), color, thick, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * corner), color, thick, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Head-shake (snooze gesture) detector
# ---------------------------------------------------------------------------
def detect_head_shake(yaw_history):
    """Return True if recent yaw history looks like a deliberate head shake:
    a wide left-right sweep with at least two direction reversals in ~1s."""
    if len(yaw_history) < 6:
        return False
    now = yaw_history[-1][0]
    recent = [y for (t, y) in yaw_history if now - t <= 1.2]
    if len(recent) < 6:
        return False
    if max(recent) - min(recent) < 25:   # needs a wide sweep
        return False
    reversals = 0
    for i in range(2, len(recent)):
        d1 = recent[i - 1] - recent[i - 2]
        d2 = recent[i] - recent[i - 1]
        if d1 * d2 < 0 and abs(d2) > 3:
            reversals += 1
    return reversals >= 2


# ---------------------------------------------------------------------------
# Startup calibration
# ---------------------------------------------------------------------------
def calibrate_ear(cap, face_mesh, seconds=3.0):
    """Sample the user's open-eye EAR for a few seconds and derive a personal
    closed-eye threshold. Returns a threshold, or None to keep the default.

    [S] skips calibration, [Q] quits.
    """
    samples = []
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        remaining = seconds - (time.time() - start)

        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark
            ear_l = eye_aspect_ratio(lms, LEFT_EYE_INDICES, w, h)
            ear_r = eye_aspect_ratio(lms, RIGHT_EYE_INDICES, w, h)
            samples.append((ear_l + ear_r) / 2.0)

        cv2.rectangle(frame, (0, h // 2 - 60), (w, h // 2 + 60), (20, 20, 20), -1)
        cv2.putText(frame, "CALIBRATING - keep your eyes open",
                    (w // 2 - 320, h // 2 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                    (0, 220, 60), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{max(0.0, remaining):.1f}s   [S] skip  [Q] quit",
                    (w // 2 - 200, h // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imshow("Sleep Alarm Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return "quit"
        if key == ord('s'):
            return None
        if remaining <= 0:
            break

    if len(samples) < 5:
        print("[CALIBRATION] Not enough samples - using default threshold.")
        return None
    baseline = float(np.median(samples))
    # closed eyes sit well below the open baseline; 70% is a good margin
    threshold = max(0.15, min(0.30, baseline * 0.70))
    print(f"[CALIBRATION] Open-eye EAR baseline {baseline:.3f} "
          f"-> threshold {threshold:.3f}")
    return threshold


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    global EAR_THRESHOLD
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # --- personalise the closed-eye threshold to this user ---
    calibrated = calibrate_ear(cap, face_mesh)
    if calibrated == "quit":
        cap.release()
        face_mesh.close()
        cv2.destroyAllWindows()
        return
    if calibrated is not None:
        EAR_THRESHOLD = calibrated

    # --- state ---
    eye_closed_start = None
    prev_eyes_closed = False
    droop_start = None

    alarm_active = False
    alarm_start = None
    last_voice_time = 0.0
    snooze_until = 0.0

    yawn_active = False
    yawn_start = None
    blink_times = deque()   # timestamps of completed blinks
    yawn_times = deque()    # timestamps of completed yawns
    yaw_history = deque(maxlen=60)

    print("=" * 62)
    print("  Smart Sleep Detector")
    print("  [Q] quit   [S] snooze   (or shake your head to snooze an alarm)")
    print(f"  Alarm on: {EYE_CLOSED_SECONDS}s eyes closed / {HEAD_DROOP_SECONDS}s head droop"
          f" / score >= {SCORE_ALARM_THRESHOLD}")
    print(f"  Voice alerts: {'on' if TTS_AVAILABLE else 'off'}   "
          f"Audio: {'on' if PYGAME_AVAILABLE else 'off'}")
    print("=" * 62)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            now = time.time()

            ear_avg = 1.0
            mar = 0.0
            pitch = 0.0
            eye_elapsed = 0.0
            droop_elapsed = 0.0
            face_detected = False
            snoozed = now < snooze_until

            if results.multi_face_landmarks:
                face_detected = True
                lms = results.multi_face_landmarks[0].landmark

                # ---- eyes ----
                ear_l = eye_aspect_ratio(lms, LEFT_EYE_INDICES, w, h)
                ear_r = eye_aspect_ratio(lms, RIGHT_EYE_INDICES, w, h)
                ear_avg = (ear_l + ear_r) / 2.0
                eyes_closed = ear_avg < EAR_THRESHOLD

                if eyes_closed:
                    if eye_closed_start is None:
                        eye_closed_start = now
                    eye_elapsed = now - eye_closed_start
                else:
                    # transition closed -> open: was it a blink?
                    if prev_eyes_closed and eye_closed_start is not None:
                        if now - eye_closed_start <= BLINK_MAX_SECONDS:
                            blink_times.append(now)
                    eye_closed_start = None
                prev_eyes_closed = eyes_closed

                # ---- mouth / yawns ----
                mar = mouth_aspect_ratio(lms, w, h)
                if mar > MAR_THRESHOLD:
                    if yawn_start is None:
                        yawn_start = now
                    if not yawn_active and now - yawn_start >= YAWN_MIN_SECONDS:
                        yawn_active = True
                        yawn_times.append(now)
                else:
                    yawn_start = None
                    yawn_active = False

                # ---- head pose ----
                pose = head_pose_angles(lms, w, h)
                if pose is not None:
                    pitch, yaw, _ = pose
                    yaw_history.append((now, yaw))
                    if abs(pitch) > HEAD_DROOP_DEG:
                        if droop_start is None:
                            droop_start = now
                        droop_elapsed = now - droop_start
                    else:
                        droop_start = None
            else:
                # No face: reset transient timers (but keep rolling metrics)
                eye_closed_start = None
                prev_eyes_closed = False
                droop_start = None
                yawn_start = None
                yawn_active = False

            # ---- prune rolling-window metrics ----
            while blink_times and now - blink_times[0] > METRIC_WINDOW_SECONDS:
                blink_times.popleft()
            while yawn_times and now - yawn_times[0] > METRIC_WINDOW_SECONDS:
                yawn_times.popleft()
            blink_rate = len(blink_times)   # blinks in the last ~minute
            yawn_rate = len(yawn_times)

            score = compute_drowsiness_score(eye_elapsed, yawn_rate, droop_elapsed, blink_rate)

            # ---- decide alarm ----
            should_alarm = face_detected and not snoozed and (
                eye_elapsed >= EYE_CLOSED_SECONDS
                or droop_elapsed >= HEAD_DROOP_SECONDS
                or score >= SCORE_ALARM_THRESHOLD
            )

            if should_alarm and not alarm_active:
                alarm_active = True
                alarm_start = now
                last_voice_time = now
                play_alarm(volume=0.4)
                speak("Wake up!")
            elif alarm_active and (snoozed or not face_detected or (
                    eye_elapsed < EYE_CLOSED_SECONDS
                    and droop_elapsed < HEAD_DROOP_SECONDS
                    and score < SCORE_ALARM_THRESHOLD)):
                alarm_active = False
                alarm_start = None
                stop_alarm()

            # ---- escalate a live alarm (volume ramp + repeated voice) ----
            if alarm_active and alarm_start is not None:
                ramp = min((now - alarm_start) / ALARM_RAMP_SECONDS, 1.0)
                set_alarm_volume(0.4 + 0.6 * ramp)
                if now - last_voice_time >= VOICE_REPEAT_SECONDS:
                    speak("Wake up!")
                    last_voice_time = now

            # ---- head-shake to snooze (only meaningful during an alarm) ----
            if alarm_active and detect_head_shake(yaw_history):
                snooze_until = now + SNOOZE_SECONDS
                alarm_active = False
                alarm_start = None
                stop_alarm()
                eye_closed_start = None
                droop_start = None
                yaw_history.clear()
                print("[SNOOZE] Head shake detected - snoozing.")

            # ---- draw ----
            if face_detected:
                face_color = (0, 0, 220) if alarm_active else (0, 220, 60)
                draw_face_box(frame, results.multi_face_landmarks[0].landmark, face_color, w, h)

            stats = {
                "score": score, "ear": ear_avg, "mar": mar, "pitch": pitch,
                "eye_elapsed": eye_elapsed, "blink_rate": blink_rate,
                "yawn_count": yawn_rate, "snoozed": snoozed,
                "snooze_left": max(0.0, snooze_until - now),
            }
            frame = draw_status_overlay(frame, alarm_active, stats)
            if not face_detected:
                cv2.putText(frame, "No face detected", (w // 2 - 130, h // 2),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 180, 255), 2, cv2.LINE_AA)

            cv2.imshow("Sleep Alarm Detector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                snooze_until = now + SNOOZE_SECONDS
                if alarm_active:
                    alarm_active = False
                    alarm_start = None
                    stop_alarm()
                eye_closed_start = None
                droop_start = None
                print("[SNOOZE] Manual snooze.")
    finally:
        stop_alarm()
        cap.release()
        face_mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

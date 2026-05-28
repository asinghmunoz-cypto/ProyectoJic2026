import cv2
import numpy as np
import mediapipe as mp
import time

# =====================================================
# Retinex 
# =====================================================

BRIGHTNESS_THRESHOLD = 50

EAR_THRESHOLD = 0.22
CONSEC_FRAMES = 15

sleep_counter = 0

eyes_closed_start = None

# =====================================================
# MEDIAPIPE
# =====================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# =====================================================
# EAR
# =====================================================

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(eye_points):

    vertical1 = euclidean(eye_points[1], eye_points[5])
    vertical2 = euclidean(eye_points[2], eye_points[4])

    horizontal = euclidean(eye_points[0], eye_points[3])

    return (vertical1 + vertical2) / (2.0 * horizontal)

# =====================================================
# RETINEX
# =====================================================

def retinex_enhancement(frame):

    img = frame.astype(np.float32) / 255.0

    illumination = cv2.GaussianBlur(img, (0, 0), sigmaX=15)

    illumination = np.maximum(illumination, 0.01)

    reflectance = img / illumination

    reflectance = cv2.bilateralFilter(
        reflectance.astype(np.float32),
        d=5,
        sigmaColor=50,
        sigmaSpace=50
    )

    enhanced = illumination * reflectance

    enhanced = cv2.normalize(
        enhanced,
        None,
        0,
        1,
        cv2.NORM_MINMAX
    )

    enhanced = np.clip(enhanced * 1.2, 0, 1)

    enhanced = (enhanced * 255).astype(np.uint8)

    return enhanced

# =====================================================
# BRIGHTNESS
# =====================================================

def compute_brightness(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return np.mean(gray)

# =====================================================
# DETECTION
# =====================================================

def detect_drowsiness(frame):

    global sleep_counter, eyes_closed_start

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            h, w, _ = frame.shape

            left_eye = []
            right_eye = []

            for idx in LEFT_EYE:

                point = face_landmarks.landmark[idx]

                left_eye.append(
                    (int(point.x * w), int(point.y * h))
                )

            for idx in RIGHT_EYE:

                point = face_landmarks.landmark[idx]

                right_eye.append(
                    (int(point.x * w), int(point.y * h))
                )

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)

            ear = (left_ear + right_ear) / 2.0

            # dibujar puntos

            for p in left_eye + right_eye:
                cv2.circle(frame, p, 2, (0, 255, 0), -1)

            # ============================================
            # DETECCIÓN POR TIEMPO
            # ============================================

            if ear < EAR_THRESHOLD:

                if eyes_closed_start is None:

                    eyes_closed_start = time.time()

                closed_time = time.time() - eyes_closed_start

            else:

                eyes_closed_start = None

                closed_time = 0

            if closed_time >= 2:

                cv2.putText(
                    frame,
                    "SOMNOLENCIA",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Closed: {closed_time:.1f}s",
                (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 255),
                2
            )

    return frame

# =====================================================
# CAMERA
# =====================================================

cap = cv2.VideoCapture("Prueba 39s.mp4")

prev_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # espejo
    frame = cv2.flip(frame, 1)

    # ============================================
    # brillo
    # ============================================

    brightness = compute_brightness(frame)

    low_light = False

    if brightness < BRIGHTNESS_THRESHOLD:

        low_light = True

        frame = retinex_enhancement(frame)

    # ============================================
    # detección
    # ============================================

    frame = detect_drowsiness(frame)

    # ============================================
    # fps
    # ============================================

    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (30, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Brightness: {brightness:.1f}",
        (30, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    if low_light:

        cv2.putText(
            frame,
            "RETINEX ON",
            (30, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Detector Somnolencia", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()
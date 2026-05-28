import cv2
import numpy as np
import mediapipe as mp
import time

# =====================================================
# RETINEX + EXPOSURE
# =====================================================

BRIGHTNESS_THRESHOLD = 50

EAR_THRESHOLD = 0.22

DROWSINESS_TIME = 2.0

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

    illumination = cv2.GaussianBlur(
        img,
        (0, 0),
        sigmaX=15
    )

    illumination = np.maximum(illumination, 0.01)

    reflectance = img / illumination

    reflectance = cv2.bilateralFilter(
        np.float32(reflectance),
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
# DETECCIÓN
# =====================================================

def detect_drowsiness(frame):

    global eyes_closed_start

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    closed_time = 0

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            h, w, _ = frame.shape

            left_eye = []
            right_eye = []

            # ojo izquierdo

            for idx in LEFT_EYE:

                point = face_landmarks.landmark[idx]

                left_eye.append(
                    (int(point.x * w), int(point.y * h))
                )

            # ojo derecho

            for idx in RIGHT_EYE:

                point = face_landmarks.landmark[idx]

                right_eye.append(
                    (int(point.x * w), int(point.y * h))
                )

            # EAR

            left_ear = calculate_ear(left_eye)

            right_ear = calculate_ear(right_eye)

            ear = (left_ear + right_ear) / 2.0

            # landmarks

            for p in left_eye + right_eye:

                cv2.circle(
                    frame,
                    p,
                    1,
                    (0, 255, 0),
                    -1
                )

            # detección temporal

            if ear < EAR_THRESHOLD:

                if eyes_closed_start is None:

                    eyes_closed_start = time.time()

                closed_time = (
                    time.time() - eyes_closed_start
                )

            else:

                eyes_closed_start = None

                closed_time = 0

            # alerta

            if closed_time >= DROWSINESS_TIME:

                cv2.putText(
                    frame,
                    "SOMNOLENCIA",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1
                )

            # EAR

            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1
            )

    return frame

# =====================================================
# CAMERA
# =====================================================

cap = cv2.VideoCapture(0)

# =====================================================
# EXPOSURE SETTINGS
# =====================================================

cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

cap.set(cv2.CAP_PROP_EXPOSURE, -5)

cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)

# =====================================================
# FPS
# =====================================================

prev_time = time.time()

# =====================================================
# MAIN LOOP
# =====================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # espejo

    frame = cv2.flip(frame, 1)

    # resolución balanceada

    frame = cv2.resize(frame, (640, 360))

    # =================================================
    # BRILLO
    # =================================================

    brightness = compute_brightness(frame)

    low_light = False

    # =================================================
    # RETINEX SOLO EN BAJA LUZ
    # =================================================

    if brightness < BRIGHTNESS_THRESHOLD:

        low_light = True

        frame = retinex_enhancement(frame)

    # =================================================
    # DETECCIÓN
    # =================================================

    frame = detect_drowsiness(frame)

    # =================================================
    # FPS
    # =================================================

    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    # =================================================
    # TEXTO
    # =================================================

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"Brightness: {brightness:.1f}",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    if low_light:

        cv2.putText(
            frame,
            "RETINEX",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    # =================================================
    # MOSTRAR
    # =================================================

    cv2.imshow(
        "Detector Somnolencia",
        frame
    )

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

# =====================================================
# FINAL
# =====================================================

cap.release()

cv2.destroyAllWindows()
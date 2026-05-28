import cv2
import numpy as np
import mediapipe as mp
import time

# =====================================================
# RETINEX + CLAHE VERSION
# =====================================================
#
# CAMBIOS RESPECTO A SOLO RETINEX:
#
# 1. Se agregó CLAHE
#    → mejora contraste local
#    → ayuda a resaltar ojos/cara
#    → mejora landmarks en baja luz
#
# 2. Se corrigieron formatos uint8
#    → evita crashes de OpenCV
#
# 3. Se mantuvo:
#    - Retinex
#    - EAR
#    - detección por tiempo
#    - mejora dinámica por brillo
#
# =====================================================

# =====================================================
# CONFIGURACIÓN
# =====================================================

BRIGHTNESS_THRESHOLD = 50

EAR_THRESHOLD = 0.22

# tiempo mínimo con ojos cerrados
DROWSINESS_TIME = 2.0

# =====================================================
# VARIABLES
# =====================================================

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

    ear = (vertical1 + vertical2) / (2.0 * horizontal)

    return ear

# =====================================================
# RETINEX + CLAHE
# =====================================================

def retinex_enhancement(frame):

    # -------------------------------------------------
    # NORMALIZACIÓN
    # -------------------------------------------------
    # Convierte imagen:
    # 0-255 → 0.0-1.0
    # para operaciones matemáticas estables
    # -------------------------------------------------

    img = frame.astype(np.float32) / 255.0

    # -------------------------------------------------
    # ESTIMACIÓN DE ILUMINACIÓN
    # -------------------------------------------------
    # Gaussian Blur modela iluminación ambiente
    # -------------------------------------------------

    illumination = cv2.GaussianBlur(
        img,
        (0, 0),
        sigmaX=15
    )

    # evitar división por cero

    illumination = np.maximum(illumination, 0.01)

    # -------------------------------------------------
    # REFLECTANCIA
    # -------------------------------------------------
    # separa detalles reales de la iluminación
    # -------------------------------------------------

    reflectance = img / illumination

    # -------------------------------------------------
    # FILTRO EDGE-PRESERVING
    # -------------------------------------------------
    # reduce ruido sin destruir bordes
    # -------------------------------------------------

    reflectance = cv2.bilateralFilter(
        np.float32(reflectance),
        d=5,
        sigmaColor=50,
        sigmaSpace=50
    )

    # -------------------------------------------------
    # RECOMBINAR
    # -------------------------------------------------

    enhanced = illumination * reflectance

    # -------------------------------------------------
    # NORMALIZAR CONTRASTE
    # -------------------------------------------------

    enhanced = cv2.normalize(
        enhanced,
        None,
        0,
        1,
        cv2.NORM_MINMAX
    )

    # -------------------------------------------------
    # AUMENTO SUAVE DE BRILLO
    # -------------------------------------------------

    enhanced = np.clip(enhanced * 1.2, 0, 1)

    # -------------------------------------------------
    # CONVERTIR A uint8
    # -------------------------------------------------
    # NECESARIO para OpenCV + CLAHE
    # -------------------------------------------------

    enhanced = (enhanced * 255).astype(np.uint8)

    # =================================================
    # NUEVO:
    # CLAHE
    # =================================================
    #
    # SOLO EXISTE EN ESTA VERSIÓN
    #
    # Mejora contraste local:
    # - ojos
    # - párpados
    # - bordes faciales
    #
    # MUCHÍSIMO mejor para baja luz
    #
    # =================================================

    # convertir a espacio LAB

    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)

    # separar canales

    l, a, b = cv2.split(lab)

    # crear CLAHE

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    # aplicar SOLO al canal de iluminación

    l = clahe.apply(l.astype(np.uint8))

    # recombinar canales

    lab = cv2.merge((l, a, b))

    # volver a BGR

    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return enhanced

# =====================================================
# BRILLO
# =====================================================

def compute_brightness(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return np.mean(gray)

# =====================================================
# DETECCIÓN DE SOMNOLENCIA
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

            # -------------------------------------------------
            # OJO IZQUIERDO
            # -------------------------------------------------

            for idx in LEFT_EYE:

                point = face_landmarks.landmark[idx]

                left_eye.append(
                    (int(point.x * w), int(point.y * h))
                )

            # -------------------------------------------------
            # OJO DERECHO
            # -------------------------------------------------

            for idx in RIGHT_EYE:

                point = face_landmarks.landmark[idx]

                right_eye.append(
                    (int(point.x * w), int(point.y * h))
                )

            # -------------------------------------------------
            # EAR
            # -------------------------------------------------

            left_ear = calculate_ear(left_eye)

            right_ear = calculate_ear(right_eye)

            ear = (left_ear + right_ear) / 2.0

            # -------------------------------------------------
            # DIBUJAR LANDMARKS
            # -------------------------------------------------

            for p in left_eye + right_eye:

                cv2.circle(
                    frame,
                    p,
                    2,
                    (0, 255, 0),
                    -1
                )

            # -------------------------------------------------
            # DETECCIÓN POR TIEMPO
            # -------------------------------------------------

            if ear < EAR_THRESHOLD:

                if eyes_closed_start is None:

                    eyes_closed_start = time.time()

                closed_time = (
                    time.time() - eyes_closed_start
                )

            else:

                eyes_closed_start = None

                closed_time = 0

            # -------------------------------------------------
            # ALERTA
            # -------------------------------------------------

            if closed_time >= DROWSINESS_TIME:

                cv2.putText(
                    frame,
                    "SOMNOLENCIA",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

            # -------------------------------------------------
            # MOSTRAR EAR
            # -------------------------------------------------

            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            # -------------------------------------------------
            # MOSTRAR TIEMPO CERRADO
            # -------------------------------------------------

            cv2.putText(
                frame,
                f"Closed: {closed_time:.1f}s",
                (30, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 255),
                2
            )

    return frame

# =====================================================
# CÁMARA
# =====================================================

cap = cv2.VideoCapture(0)

# =====================================================
# NUEVO:
# exposición manual
# =====================================================
#
# ayuda muchísimo en baja luz
#
# =====================================================

cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)

cap.set(cv2.CAP_PROP_EXPOSURE, -4)

prev_time = time.time()

# =====================================================
# LOOP PRINCIPAL
# =====================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # espejo

    frame = cv2.flip(frame, 1)

    # =================================================
    # NUEVO:
    # resolución reducida
    # =================================================
    #
    # mejora rendimiento
    # reduce ruido
    #
    # =================================================

    frame = cv2.resize(frame, (640, 480))

    # =================================================
    # BRILLO
    # =================================================

    brightness = compute_brightness(frame)

    low_light = False

    # =================================================
    # RETINEX SOLO SI HAY BAJA LUZ
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
    # TEXTO FPS
    # =================================================

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (30, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # =================================================
    # TEXTO BRILLO
    # =================================================

    cv2.putText(
        frame,
        f"Brightness: {brightness:.1f}",
        (30, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # =================================================
    # RETINEX ACTIVO
    # =================================================

    if low_light:

        cv2.putText(
            frame,
            "RETINEX + CLAHE",
            (30, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
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
# FINALIZAR
# =====================================================

cap.release()

cv2.destroyAllWindows()
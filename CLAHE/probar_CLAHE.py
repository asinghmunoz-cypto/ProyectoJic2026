import os
import sys
import cv2
import time
import numpy as np
import mediapipe as mp

# Permite importar el paquete metricas_ofc desde la carpeta padre de CLAHE
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metodo_CLAHE import aplicar_CLAHE

from metricas_ofc.ear import (
    calcular_EAR,
    LEFT_EYE,
    RIGHT_EYE
)

from metricas_ofc.mar import (
    calculate_mar,
    MOUTH
)

from metricas_ofc.psnr import calcular_psnr
from metricas_ofc.ssim import calcular_ssim
from metricas_ofc.iluminacion import evaluar_iluminacion
from metricas_ofc.fps import actualizar_fps


# ==================================================
# CONFIGURACIÓN
# ==================================================

VIDEO_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        '..',
        'data',
        'prueba_39s.mp4'
    )
)

# ==================================================
# MEDIAPIPE
# ==================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==================================================
# VIDEO
# ==================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"No se pudo abrir el video: {VIDEO_PATH}"
    )

# ==================================================
# MÉTRICAS
# ==================================================

ear_lista = []
mar_lista = []

psnr_lista = []
ssim_lista = []

brightness_lista = []
dark_ratio_lista = []
std_dev_lista = []

fps_lista = []

fps = 0
prev_time = time.time()

# ==================================================
# BUCLE PRINCIPAL
# ==================================================

while True:

    ret, frame_original = cap.read()

    if not ret:
        break

    h, w = frame_original.shape[:2]

    rgb = cv2.cvtColor(
        frame_original,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(rgb)

    frame_mejorado = frame_original.copy()

    if results.multi_face_landmarks:

        face = results.multi_face_landmarks[0]

        puntos = np.array([
            (
                int(lm.x * w),
                int(lm.y * h)
            )
            for lm in face.landmark
        ])

        x, y, ww, hh = cv2.boundingRect(puntos)

        margen = 15

        x = max(0, x - margen)
        y = max(0, y - margen)

        ww = min(ww + 2*margen, w - x)
        hh = min(hh + 2*margen, h - y)

        roi = frame_original[
            y:y + hh,
            x:x + ww
        ]

        if roi.size > 0:

            roi_clahe = aplicar_CLAHE(roi)

            frame_mejorado[
                y:y + hh,
                x:x + ww
            ] = roi_clahe

        # ==========================================
        # EAR
        # ==========================================

        left_eye = []

        for idx in LEFT_EYE:

            px = int(
                face.landmark[idx].x * w
            )

            py = int(
                face.landmark[idx].y * h
            )

            left_eye.append((px, py))

        right_eye = []

        for idx in RIGHT_EYE:

            px = int(
                face.landmark[idx].x * w
            )

            py = int(
                face.landmark[idx].y * h
            )

            right_eye.append((px, py))

        ear_left = calcular_EAR(left_eye)
        ear_right = calcular_EAR(right_eye)

        ear = (ear_left + ear_right) / 2

        ear_lista.append(ear)

        # ==========================================
        # MAR
        # ==========================================

        mouth_points = []

        for idx in MOUTH:

            px = int(
                face.landmark[idx].x * w
            )

            py = int(
                face.landmark[idx].y * h
            )

            mouth_points.append((px, py))

        mar, _, _, _, _ = calculate_mar(
            mouth_points
        )

        mar_lista.append(mar)

        cv2.putText(
            frame_mejorado,
            f"EAR: {ear:.3f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame_mejorado,
            f"MAR: {mar:.3f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # ==========================================
    # ILUMINACIÓN
    # ==========================================

    info_ilum = evaluar_iluminacion(
        frame_mejorado
    )

    brightness_lista.append(
        info_ilum["brightness"]
    )

    dark_ratio_lista.append(
        info_ilum["dark_ratio"]
    )

    std_dev_lista.append(
        info_ilum["std_dev"]
    )

    # ==========================================
    # PSNR
    # ==========================================

    psnr = calcular_psnr(
        frame_original,
        frame_mejorado
    )

    if psnr != float("inf"):
        psnr_lista.append(psnr)

    # ==========================================
    # SSIM
    # ==========================================

    ssim = calcular_ssim(
        frame_original,
        frame_mejorado
    )

    ssim_lista.append(ssim)

    # ==========================================
    # FPS
    # ==========================================

    fps, prev_time = actualizar_fps(
        fps,
        prev_time
    )

    fps_lista.append(fps)

    cv2.putText(
        frame_mejorado,
        f"FPS: {fps:.1f}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    comparacion = np.hstack(
        (
            frame_original,
            frame_mejorado
        )
    )

    cv2.imshow(
        "CLAHE vs Original",
        comparacion
    )

    tecla = cv2.waitKey(1)

    if tecla == 27:
        break

# ==================================================
# CIERRE
# ==================================================

cap.release()
cv2.destroyAllWindows()

# ==================================================
# RESULTADOS
# ==================================================

print("\n==============================")
print("RESULTADOS CLAHE")
print("==============================")

if ear_lista:
    print(
        f"EAR promedio: {np.mean(ear_lista):.4f}"
    )

if mar_lista:
    print(
        f"MAR promedio: {np.mean(mar_lista):.4f}"
    )

if brightness_lista:
    print(
        f"Brightness promedio: "
        f"{np.mean(brightness_lista):.2f}"
    )

if dark_ratio_lista:
    print(
        f"Dark Ratio promedio: "
        f"{np.mean(dark_ratio_lista):.4f}"
    )

if std_dev_lista:
    print(
        f"Std Dev promedio: "
        f"{np.mean(std_dev_lista):.2f}"
    )

if psnr_lista:
    print(
        f"PSNR promedio: "
        f"{np.mean(psnr_lista):.2f}"
    )

if ssim_lista:
    print(
        f"SSIM promedio: "
        f"{np.mean(ssim_lista):.4f}"
    )

if fps_lista:
    print(
        f"FPS promedio: "
        f"{np.mean(fps_lista):.2f}"
    )
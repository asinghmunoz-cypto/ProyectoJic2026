import os
import csv
import cv2
import mediapipe as mp
import math

# ==================================================
# CARPETAS A EVALUAR
# ==================================================

DATASETS = {
    "40": "dataset1/ojoscerrados_B40",
    "25": "dataset1/ojoscerrados_B25",
    "15": "dataset1/ojoscerrados_B15",
    "10": "dataset1/ojoscerrados_B10",
    "5":  "dataset1/ojoscerrados_B5"
}

CSV_SALIDA = "comparacion_EAR_iluminacion.csv"

ANCHO_STD = 640
ALTO_STD = 480

# ==================================================
# LANDMARKS
# ==================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ==================================================
# FUNCIONES
# ==================================================

def distancia(p1, p2):
    return math.hypot(
        p1[0] - p2[0],
        p1[1] - p2[1]
    )

def calcular_EAR(puntos):
    A = distancia(puntos[1], puntos[5])
    B = distancia(puntos[2], puntos[4])
    C = distancia(puntos[0], puntos[3])

    if C == 0:
        return 0

    return (A + B) / (2.0 * C)

# ==================================================
# RESULTADOS
# ==================================================

filas = []

mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
) as face_mesh:

    for nivel, carpeta in DATASETS.items():

        print(f"\nProcesando B{nivel}")

        total = 0
        detectadas = 0

        for nombre in sorted(os.listdir(carpeta)):

            if not nombre.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            ):
                continue

            total += 1

            ruta = os.path.join(carpeta, nombre)

            frame = cv2.imread(ruta)

            if frame is None:
                continue

            frame = cv2.resize(
                frame,
                (ANCHO_STD, ALTO_STD)
            )

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = face_mesh.process(rgb)

            if not results.multi_face_landmarks:

                filas.append({
                    "imagen": nombre,
                    "brightness": nivel,
                    "detectado": 0,
                    "EAR": ""
                })

                continue

            detectadas += 1

            face_landmarks = results.multi_face_landmarks[0]

            left_eye_points = []
            right_eye_points = []

            for idx in LEFT_EYE:

                x = int(
                    face_landmarks.landmark[idx].x *
                    ANCHO_STD
                )

                y = int(
                    face_landmarks.landmark[idx].y *
                    ALTO_STD
                )

                left_eye_points.append((x, y))

            for idx in RIGHT_EYE:

                x = int(
                    face_landmarks.landmark[idx].x *
                    ANCHO_STD
                )

                y = int(
                    face_landmarks.landmark[idx].y *
                    ALTO_STD
                )

                right_eye_points.append((x, y))

            ear_izq = calcular_EAR(left_eye_points)
            ear_der = calcular_EAR(right_eye_points)

            ear = (ear_izq + ear_der) / 2

            filas.append({
                "imagen": nombre,
                "brightness": nivel,
                "detectado": 1,
                "EAR": round(ear, 6)
            })

        print(
            f"B{nivel}: "
            f"{detectadas}/{total} detectadas"
        )

# ==================================================
# GUARDAR CSV
# ==================================================

with open(
    CSV_SALIDA,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "imagen",
            "brightness",
            "detectado",
            "EAR"
        ]
    )

    writer.writeheader()
    writer.writerows(filas)

print("\nCSV generado:")
print(CSV_SALIDA)

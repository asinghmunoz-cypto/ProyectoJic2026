import os
import csv
import cv2
import mediapipe as mp
import math
from openpyxl import Workbook

# ==================================================
# RUTAS
# ==================================================

CARPETA_ORIGEN = "dataset1/ojoscerrados_500"

CARPETA_DESTINO = "dataset1/ojoscerrados_EAR"

os.makedirs(CARPETA_DESTINO, exist_ok=True)

CARPETA_RESULTADOS = "resultados dataset1"
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

EXCEL_SALIDA = os.path.join(CARPETA_RESULTADOS, "EAR_cerrados_resultados.xlsx")
CSV_SALIDA   = os.path.join(CARPETA_RESULTADOS, "EAR_cerrados_resultados.csv")

# Tamaño estándar al que se normalizan todas las imágenes
ANCHO_STD = 640
ALTO_STD  = 480

# ==================================================
# LANDMARKS DE LOS OJOS
# ==================================================

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ==================================================
# FUNCIONES
# ==================================================

def distancia(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def calcular_EAR(puntos):
    A = distancia(puntos[1], puntos[5])
    B = distancia(puntos[2], puntos[4])
    C = distancia(puntos[0], puntos[3])
    if C == 0:
        return 0
    return (A + B) / (2.0 * C)

# ==================================================
# CREAR EXCEL Y PREPARAR CSV
# ==================================================

wb = Workbook()
ws = wb.active
ws.title = "EAR"
ws.append(["Imagen", "EAR"])

filas_csv = []

# ==================================================
# MEDIAPIPE
# ==================================================

mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
) as face_mesh:

    archivos = sorted(os.listdir(CARPETA_ORIGEN))

    total     = 0
    detectadas = 0

    for nombre in archivos:

        if not nombre.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue

        total += 1

        ruta = os.path.join(CARPETA_ORIGEN, nombre)
        frame = cv2.imread(ruta)

        if frame is None:
            print(f"No se pudo leer: {nombre}")
            continue

        # Normalizar tamaño para que el texto y los puntos queden bien
        frame = cv2.resize(frame, (ANCHO_STD, ALTO_STD))

        h, w = ALTO_STD, ANCHO_STD

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:

            print(f"{nombre} -> NO DETECTADO")

            cv2.putText(
                frame,
                "NO DETECTADO",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

            ws.append([nombre, "No detectado"])
            filas_csv.append({"imagen": nombre, "EAR": "No detectado"})

        else:

            detectadas += 1
            face_landmarks = results.multi_face_landmarks[0]

            left_eye_points  = []
            right_eye_points = []

            # Ojo izquierdo
            for idx in LEFT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                left_eye_points.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Ojo derecho
            for idx in RIGHT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                right_eye_points.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            ear_izq = calcular_EAR(left_eye_points)
            ear_der = calcular_EAR(right_eye_points)
            ear     = (ear_izq + ear_der) / 2

            print(f"{nombre} -> EAR = {ear:.6f}")

            cv2.putText(
                frame,
                f"EAR: {ear:.4f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

            ws.append([nombre, round(ear, 6)])
            filas_csv.append({"imagen": nombre, "EAR": round(ear, 6)})

        # Guardar imagen con EAR escrito
        ruta_salida = os.path.join(CARPETA_DESTINO, nombre)
        cv2.imwrite(ruta_salida, frame)

        # Mostrar en pantalla (ya está a 640x480)
        cv2.imshow("EAR - Ojos Cerrados", frame)

        tecla = cv2.waitKey(0)
        if tecla == 27:  # ESC
            break

cv2.destroyAllWindows()

# ==================================================
# GUARDAR EXCEL
# ==================================================

wb.save(EXCEL_SALIDA)

with open(CSV_SALIDA, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["imagen", "EAR"])
    writer.writeheader()
    writer.writerows(filas_csv)

print("\n====================================")
print("PROCESO FINALIZADO")
print("====================================")
print(f"Imágenes procesadas : {total}")
print(f"Rostros detectados  : {detectadas}")
print(f"Excel guardado en   : {EXCEL_SALIDA}")
print(f"CSV guardado en     : {CSV_SALIDA}")
print(f"Imágenes guardadas en: {CARPETA_DESTINO}")

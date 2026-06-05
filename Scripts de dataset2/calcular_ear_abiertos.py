import csv
import cv2
import mediapipe as mp
import math
import os
from openpyxl import Workbook

# ==========================================
# RUTAS
# ==========================================

CARPETA_ORIGEN = r"C:\Users\MovilCity\Downloads\Ojos_abiertos500"

CARPETA_DESTINO = r"C:\Users\MovilCity\Downloads\Ojos_abiertosEAR"

os.makedirs(CARPETA_DESTINO, exist_ok=True)

EXCEL_SALIDA = os.path.join(CARPETA_DESTINO, "EAR_resultados.xlsx")
CSV_SALIDA   = os.path.join(CARPETA_DESTINO, "EAR_resultados.csv")

# ==========================================
# LANDMARKS DE LOS OJOS
# ==========================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ==========================================
# FUNCIONES
# ==========================================

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

# ==========================================
# CREAR EXCEL
# ==========================================

wb = Workbook()
ws = wb.active
ws.title = "EAR"
ws.append(["Imagen", "EAR"])

filas_csv = []

# ==========================================
# MEDIAPIPE
# ==========================================

mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
) as face_mesh:

    archivos = os.listdir(CARPETA_ORIGEN)

    total = 0
    detectadas = 0

    for archivo in archivos:

        if not archivo.lower().endswith(
            ('.jpg', '.jpeg', '.png', '.bmp')
        ):
            continue

        total += 1

        ruta_imagen = os.path.join(
            CARPETA_ORIGEN,
            archivo
        )

        imagen = cv2.imread(ruta_imagen)

        if imagen is None:
            print(f"No se pudo leer: {archivo}")
            continue

        h, w, _ = imagen.shape

        rgb = cv2.cvtColor(
            imagen,
            cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:

            print(f"Rostro no detectado: {archivo}")

            ws.append([archivo, "No detectado"])
            filas_csv.append({"imagen": archivo, "EAR": "No detectado"})

            continue

        detectadas += 1

        face_landmarks = results.multi_face_landmarks[0]

        left_eye_points = []
        right_eye_points = []

        # Ojo izquierdo
        for idx in LEFT_EYE:

            x = int(
                face_landmarks.landmark[idx].x * w
            )

            y = int(
                face_landmarks.landmark[idx].y * h
            )

            left_eye_points.append((x, y))

        # Ojo derecho
        for idx in RIGHT_EYE:

            x = int(
                face_landmarks.landmark[idx].x * w
            )

            y = int(
                face_landmarks.landmark[idx].y * h
            )

            right_eye_points.append((x, y))

        ear_izq = calcular_EAR(left_eye_points)
        ear_der = calcular_EAR(right_eye_points)

        ear = (ear_izq + ear_der) / 2

        # ==================================
        # ESCRIBIR EAR EN LA IMAGEN
        # ==================================

        cv2.putText(
            imagen,
            f"EAR: {ear:.4f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        # ==================================
        # GUARDAR IMAGEN MODIFICADA
        # ==================================

        ruta_salida = os.path.join(
            CARPETA_DESTINO,
            archivo
        )

        cv2.imwrite(
            ruta_salida,
            imagen
        )

        # ==================================
        # AGREGAR AL EXCEL
        # ==================================

        ws.append([archivo, round(ear, 6)])
        filas_csv.append({"imagen": archivo, "EAR": round(ear, 6)})

        print(
            f"{archivo} -> EAR = {ear:.6f}"
        )

# ==========================================
# GUARDAR EXCEL
# ==========================================

wb.save(EXCEL_SALIDA)

with open(CSV_SALIDA, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["imagen", "EAR"])
    writer.writeheader()
    writer.writerows(filas_csv)

print("\n====================================")
print("PROCESO FINALIZADO")
print("====================================")
print(f"Imágenes procesadas: {total}")
print(f"Rostros detectados: {detectadas}")
print(f"Excel guardado en : {EXCEL_SALIDA}")
print(f"CSV guardado en   : {CSV_SALIDA}")
print(f"Imágenes guardadas en: {CARPETA_DESTINO}")
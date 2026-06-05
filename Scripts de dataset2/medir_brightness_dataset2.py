import os
import cv2
import numpy as np
import csv

# ==========================================
# CAMBIAR RUTA
# ==========================================

script_dir = os.path.dirname(os.path.abspath(__file__))
CARPETA = os.path.join(script_dir, "..", "dataset2", "Ojos_abiertos500")

CSV_SALIDA = "brightness_dataset2.csv"

# ==========================================
# RESULTADOS
# ==========================================

resultados = []

for nombre in sorted(os.listdir(CARPETA)):

    # Ignorar archivos que no sean imágenes
    if not nombre.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp")
    ):
        continue

    ruta = os.path.join(CARPETA, nombre)

    img = cv2.imread(ruta)

    # Saltar si la imagen no pudo cargarse
    if img is None:
        continue

    # Convertir a escala de grises para calcular brillo perceptual
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # El brillo se estima como el promedio de intensidad de píxeles (0–255)
    # Un valor alto indica imagen clara; uno bajo indica imagen oscura
    brightness = float(np.mean(gray))

    resultados.append({
        "imagen": nombre,
        "brightness": round(brightness, 2)
    })

# ==========================================
# GUARDAR CSV
# ==========================================

with open(
    CSV_SALIDA,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=["imagen", "brightness"]
    )

    writer.writeheader()
    writer.writerows(resultados)

# ==========================================
# ESTADÍSTICAS
# ==========================================

valores = [r["brightness"] for r in resultados]

print("\n================================")
print("RESULTADOS")
print("================================")

print(f"Cantidad imágenes : {len(valores)}")
print(f"Brightness mínimo : {min(valores):.2f}")
print(f"Brightness máximo : {max(valores):.2f}")
print(f"Brightness medio  : {np.mean(valores):.2f}")

print("\nCSV generado:")
print(CSV_SALIDA)

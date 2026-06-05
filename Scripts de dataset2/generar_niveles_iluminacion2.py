import os
import cv2
import numpy as np

# ==========================================
# CONFIGURACIÓN
# ==========================================

script_dir = os.path.dirname(os.path.abspath(__file__))
CARPETA_ORIGEN = os.path.join(script_dir, "..", "dataset2", "Ojos_abiertos500")

NIVELES = [40, 25, 15, 10, 5]

# ==========================================
# FUNCIÓN
# ==========================================

def ajustar_brillo(img, brillo_objetivo):
    """
    Reescala linealmente los píxeles de una imagen BGR para que su canal
    gris tenga un promedio aproximado de `brillo_objetivo` (0–255).

    El factor de escala se calcula como objetivo / actual, lo que equivale
    a una transformación lineal sin desplazamiento de punto de negro.
    Los valores se recortan a [0, 255] para evitar saturación.

    Parámetros
    ----------
    img : np.ndarray   Imagen BGR leída con cv2.imread.
    brillo_objetivo : float  Nivel de brillo medio deseado (ej. 5, 10, 25…)

    Retorna
    -------
    np.ndarray  Imagen con brillo ajustado, dtype uint8.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brillo_actual = np.mean(gray)

    # Evitar división por cero en imágenes completamente negras
    if brillo_actual < 1:
        return img

    # Factor multiplicativo que lleva el brillo actual al objetivo
    factor = brillo_objetivo / brillo_actual

    nueva = img.astype(np.float32) * factor

    # Recortar para mantener valores en rango válido de uint8
    nueva = np.clip(nueva, 0, 255)

    return nueva.astype(np.uint8)

# ==========================================
# GENERAR DATASETS
# ==========================================

for nivel in NIVELES:

    carpeta_destino = f"dataset2/ojosabiertos_B{nivel}"

    os.makedirs(carpeta_destino, exist_ok=True)

    print(f"\nGenerando B{nivel}")

    for nombre in os.listdir(CARPETA_ORIGEN):

        if not nombre.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ):
            continue

        ruta = os.path.join(CARPETA_ORIGEN, nombre)

        img = cv2.imread(ruta)

        if img is None:
            continue

        img_nueva = ajustar_brillo(img, nivel)

        salida = os.path.join(carpeta_destino, nombre)

        cv2.imwrite(salida, img_nueva)

print("\nFINALIZADO")
# =============================================================================
# EVALUACIÓN EXPERIMENTAL DE PSNR EN VIDEO
#
# Compara:
#   - Frame original
#   - Frame mejorado
#
# Objetivo:
# Evaluar qué tanto un algoritmo de mejora de iluminación
# introduce distorsión respecto al video original.
#
# Uso:
# Tus compañeros solo deben reemplazar la sección:
#       "ALGORITMO DE MEJORA"
#
# Compatible con:
#   - CLAHE
#   - Gamma Correction
#   - Retinex
#   - Zero-DCE
#   - etc.
# =============================================================================

import math
import cv2
import numpy as np


# =============================================================================
# PSNR
# =============================================================================
def calcular_psnr(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse < 1e-10:
        return float('inf')
    return 10.0 * math.log10(255.0 ** 2 / mse)


# =============================================================================
# VIDEO
# =============================================================================
#cap = cv2.VideoCapture("video.mp4")
# También puede usarse:
cap = cv2.VideoCapture(0)

# ── Verificar que la cámara abrió correctamente ──────────────────────────────
if not cap.isOpened():
    print("ERROR: No se pudo abrir la cámara (índice 0).")
    print("       Prueba cambiar a cv2.VideoCapture(1) o usa un archivo de video.")
    exit()

psnr_total = 0
contador = 0

while True:

    ret, frame_original = cap.read()

    if not ret:
        print("ERROR: No se pudo leer el frame. Fin del video o cámara desconectada.")
        break

    # =========================================================================
    # ALGORITMO DE MEJORA
    # =========================================================================
    # Reemplazar esta sección con cualquier algoritmo experimental
    # =========================================================================

    # EJEMPLO SIMPLE: CLAHE
    lab = cv2.cvtColor(frame_original, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_mejorado = clahe.apply(l)

    lab_mejorado = cv2.merge((l_mejorado, a, b))

    frame_mejorado = cv2.cvtColor(
        lab_mejorado,
        cv2.COLOR_LAB2BGR
    )

    # =========================================================================
    # CALCULAR PSNR
    # =========================================================================
    psnr = calcular_psnr(frame_original, frame_mejorado)

    if psnr != float('inf'):
        psnr_total += psnr
        contador += 1

    # =========================================================================
    # MOSTRAR RESULTADOS
    # =========================================================================
    texto_psnr = "inf" if psnr == float('inf') else f"{psnr:.2f} dB"

    cv2.putText(
        frame_mejorado,
        f"PSNR: {texto_psnr}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Original", frame_original)
    cv2.imshow("Mejorado", frame_mejorado)

    # Salir con Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# =============================================================================
# RESULTADO FINAL PROMEDIO
# =============================================================================
if contador > 0:

    promedio_psnr = psnr_total / contador

    print("\n===================================")
    print("RESULTADO FINAL")
    print("===================================")
    print(f"PSNR promedio: {promedio_psnr:.2f} dB")
    print("===================================")

    print("\nInterpretación:")
    print("- PSNR más alto = menor distorsión respecto al original")
    print("- > 40 dB: excelente calidad")
    print("- 30-40 dB: buena calidad")
    print("- 20-30 dB: calidad aceptable")
    print("- < 20 dB: distorsión perceptible")


cap.release()
cv2.destroyAllWindows()

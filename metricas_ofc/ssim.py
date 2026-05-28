# =============================================================================
# EVALUACIÓN EXPERIMENTAL DE SSIM EN VIDEO
#
# Compara:
#   - Frame original
#   - Frame mejorado
#
# Objetivo:
# Evaluar qué tanto un algoritmo de mejora de iluminación
# preserva la estructura original del video.
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

import cv2
import numpy as np


# =============================================================================
# SSIM
# =============================================================================
def _ssim_canal(im1, im2):

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    win = np.outer(kernel, kernel.transpose()).astype(np.float64)

    # Luminancia
    mu1 = cv2.filter2D(im1, -1, win)[5:-5, 5:-5]
    mu2 = cv2.filter2D(im2, -1, win)[5:-5, 5:-5]

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1mu2 = mu1 * mu2

    # Contraste
    sigma1_sq = cv2.filter2D(im1 ** 2, -1, win)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(im2 ** 2, -1, win)[5:-5, 5:-5] - mu2_sq

    # Estructura
    sigma12 = cv2.filter2D(im1 * im2, -1, win)[5:-5, 5:-5] - mu1mu2

    # Fórmula SSIM
    num = (2 * mu1mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    return (num / den).mean()


def calcular_ssim(a, b):

    a = a.astype(np.float64)
    b = b.astype(np.float64)

    # Imagen RGB/BGR
    if a.ndim == 3:
        return float(np.mean([
            _ssim_canal(a[:, :, c], b[:, :, c])
            for c in range(3)
        ]))

    # Escala de grises
    return float(_ssim_canal(a, b))


# =============================================================================
# DEMO STANDALONE
# Solo corre si ejecutas este archivo directamente:  python ssim.py
# Si lo importas, este bloque NO se ejecuta.
# =============================================================================
if __name__ == "__main__":

    # =========================================================================
    # VIDEO
    # =========================================================================
    #cap = cv2.VideoCapture("video.mp4")
    # También puede usarse:
    cap = cv2.VideoCapture(0)

    # ── Verificar que la cámara abrió correctamente ──────────────────────────
    if not cap.isOpened():
        print("ERROR: No se pudo abrir la cámara (índice 0).")
        print("       Prueba cambiar a cv2.VideoCapture(1) o usa un archivo de video.")
        exit()

    ssim_total = 0
    contador = 0

    while True:

        ret, frame_original = cap.read()

        if not ret:
            print("ERROR: No se pudo leer el frame. Fin del video o cámara desconectada.")
            break

        # =====================================================================
        # ALGORITMO DE MEJORA
        # =====================================================================
        # Reemplazar esta sección con cualquier algoritmo experimental
        # =====================================================================

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

        # =====================================================================
        # CALCULAR SSIM
        # =====================================================================
        ssim = calcular_ssim(frame_original, frame_mejorado)

        ssim_total += ssim
        contador += 1

        # =====================================================================
        # MOSTRAR RESULTADOS
        # =====================================================================
        cv2.putText(
            frame_mejorado,
            f"SSIM: {ssim:.4f}",
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


    # =========================================================================
    # RESULTADO FINAL PROMEDIO
    # =========================================================================
    if contador > 0:

        promedio_ssim = ssim_total / contador

        print("\n===================================")
        print("RESULTADO FINAL")
        print("===================================")
        print(f"SSIM promedio: {promedio_ssim:.4f}")
        print("===================================")

        print("\nInterpretación:")
        print("- SSIM más alto = mejor preservación estructural")


    cap.release()
    cv2.destroyAllWindows()

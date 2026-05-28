# =============================================================================
# ARCHIVO: iluminacion.py
#
# OBJETIVO DEL SCRIPT
# -----------------------------------------------------------------------------
# Este script evalúa la iluminación de un frame de video en tiempo real usando:
#
#   1. Brillo promedio (media)
#   2. Histograma de luminancia
#   3. Porcentaje de píxeles oscuros
#
# La idea es determinar si la escena tiene:
#
#   - iluminación adecuada
#   - iluminación moderada
#   - iluminación crítica
#
# para decidir posteriormente si debe activarse un algoritmo de mejora
# de iluminación (CLAHE, Gamma, LiteIE, Zero-DCE, etc.).
#
# -----------------------------------------------------------------------------
# ¿POR QUÉ USAR VARIAS MÉTRICAS?
#
# Solo usar brillo promedio puede ser engañoso.
#
# Ejemplo:
#   Una imagen puede tener:
#       - un foco muy brillante
#       - gran parte del rostro oscuro
#
# En ese caso el promedio podría parecer "normal",
# pero la detección facial seguiría fallando.
#
# Por eso también analizamos:
#
#   - distribución de luminancia (histograma)
#   - porcentaje de zonas oscuras
#
# =============================================================================

import cv2
import numpy as np


# =============================================================================
# UMBRALES
# -----------------------------------------------------------------------------
# Estos valores separan las bandas de iluminación.
#
# IMPORTANTE:
# Deben ajustarse experimentalmente con pruebas reales:
#
#   - día
#   - noche
#   - interiores
#   - luces del auto
#   - rostro parcialmente iluminado
#
# -----------------------------------------------------------------------------

# Brillo promedio bajo → iluminación crítica
UMBRAL_BRILLO_BAJO = 80

# Brillo promedio medio → iluminación moderada
##UMBRAL_BRILLO_MEDIO = 130

# Si más del 60% de los píxeles son oscuros → crítica
UMBRAL_OSCURO_CRITICO = 0.60

# Si más del 40% de los píxeles son oscuros → moderada
##UMBRAL_OSCURO_MODERADO = 0.40


# =============================================================================
# FUNCIÓN: evaluar_iluminacion
# -----------------------------------------------------------------------------
# ENTRADA:
#   frame BGR proveniente de OpenCV
#
# PROCESO:
#   1. convertir a escala de grises
#   2. calcular histograma
#   3. calcular brillo promedio
#   4. calcular desviación estándar
#   5. calcular porcentaje de píxeles oscuros
#   6. clasificar iluminación
#
# SALIDA:
#   diccionario con todas las métricas
#
# =============================================================================
def evaluar_iluminacion(frame):

    # -------------------------------------------------------------------------
    # Convertimos el frame a escala de grises.
    #
    # ¿Por qué?
    # Porque lo importante para detectar rostros y landmarks es la luminancia,
    # no el color.
    #
    # Además:
    #   - reduce procesamiento
    #   - simplifica cálculos
    #   - elimina variaciones de color innecesarias
    # -------------------------------------------------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -------------------------------------------------------------------------
    # HISTOGRAMA DE LUMINANCIA
    #
    # El histograma cuenta cuántos píxeles existen en cada intensidad:
    #
    #   0   = negro absoluto
    #   255 = blanco absoluto
    #
    # histogram[0]   → cantidad de píxeles negros
    # histogram[255] → cantidad de píxeles blancos
    #
    # Tamaño:
    #   256 posiciones → una por cada intensidad posible
    # -------------------------------------------------------------------------
    histogram = cv2.calcHist(
        [gray],     # imagen de entrada
        [0],        # canal gris
        None,       # sin máscara
        [256],      # 256 bins
        [0, 256]    # rango de intensidades
    )

    # -------------------------------------------------------------------------
    # BRILLO PROMEDIO
    #
    # Calcula la intensidad promedio de toda la imagen.
    #
    # Valores bajos:
    #   → imagen oscura
    #
    # Valores altos:
    #   → imagen brillante
    # -------------------------------------------------------------------------
    brightness = float(np.mean(gray))

    # -------------------------------------------------------------------------
    # DESVIACIÓN ESTÁNDAR
    #
    # Mide cuánto varían las intensidades.
    #
    # Baja desviación:
    #   → imagen uniforme
    #
    # Alta desviación:
    #   → mucho contraste
    #
    # Esta métrica ayuda a detectar escenas con iluminación irregular.
    # -------------------------------------------------------------------------
    std_dev = float(np.std(gray))

    # -------------------------------------------------------------------------
    # PÍXELES OSCUROS
    #
    # Tomamos el rango:
    #
    #   [0, 80]
    #
    # porque representa zonas considerablemente oscuras.
    #
    # histogram[:80]
    # significa:
    #   sumar todos los píxeles entre intensidad 0 y 79.
    # -------------------------------------------------------------------------
    low_pixels = np.sum(histogram[:80])

    # -------------------------------------------------------------------------
    # TOTAL DE PÍXELES
    #
    # gray.shape[0] → altura
    # gray.shape[1] → ancho
    # -------------------------------------------------------------------------
    total_pixels = gray.shape[0] * gray.shape[1]

    # -------------------------------------------------------------------------
    # PORCENTAJE DE OSCURIDAD
    #
    # Ejemplo:
    #
    #   0.70 = 70% de la imagen es oscura
    # -------------------------------------------------------------------------
    dark_ratio = float(low_pixels / total_pixels)

    # -------------------------------------------------------------------------
    # CLASIFICACIÓN DE ILUMINACIÓN
    #
    # La decisión usa:
    #
    #   - brillo promedio
    #   - porcentaje de oscuridad
    #
    # Esto hace el sistema más robusto que usar solo una métrica.
    # -------------------------------------------------------------------------
    if brightness < UMBRAL_BRILLO_BAJO or dark_ratio > UMBRAL_OSCURO_CRITICO:
        level = "CRITICA"

    # elif brightness < UMBRAL_BRILLO_MEDIO or dark_ratio > UMBRAL_OSCURO_MODERADO:
       # level = "MODERADA"

    else:
        level = "BIEN"

    # -------------------------------------------------------------------------
    # Retornamos todas las métricas para:
    #
    #   - visualización
    #   - análisis
    #   - calibración experimental
    # -------------------------------------------------------------------------
    return {
        "brightness": brightness,
        "std_dev": std_dev,
        "dark_ratio": dark_ratio,
        "level": level
    }


# =============================================================================
# DEMO STANDALONE
# -----------------------------------------------------------------------------
# Solo corre si ejecutas este archivo directamente con:
#   python iluminacion.py
# Si lo importas desde otro script, este bloque NO se ejecuta.
# =============================================================================
if __name__ == "__main__":

    # =========================================================================
    # CÁMARA
    # -------------------------------------------------------------------------
    # Abre la cámara principal.
    #
    # Si tienen varias cámaras:
    #
    #   0 → principal
    #   1 → secundaria
    # =========================================================================
    cap = cv2.VideoCapture(0)

    # Verificamos que la cámara abra correctamente
    if not cap.isOpened():
        print("ERROR: No se pudo abrir la cámara")
        exit()


    # =========================================================================
    # BUCLE PRINCIPAL
    # -------------------------------------------------------------------------
    # Lee frames continuamente y evalúa iluminación.
    # =========================================================================
    while True:

        # ---------------------------------------------------------------------
        # Leemos frame de cámara
        # ---------------------------------------------------------------------
        ret, frame = cap.read()

        # Si falla la lectura → salir
        if not ret:
            print("ERROR: No se pudo leer el frame")
            break

        # ---------------------------------------------------------------------
        # Evaluamos iluminación
        # ---------------------------------------------------------------------
        info = evaluar_iluminacion(frame)

        # ---------------------------------------------------------------------
        # COLOR SEGÚN SEVERIDAD
        #
        # Verde   → bien
        # Amarillo→ moderada
        # Rojo    → crítica
        # ---------------------------------------------------------------------
        if info["level"] == "BIEN":
            color = (0, 255, 0)

        elif info["level"] == "MODERADA":
            color = (0, 255, 255)

        else:
            color = (0, 0, 255)

        # ---------------------------------------------------------------------
        # Mostramos métricas en pantalla
        # ---------------------------------------------------------------------
        cv2.putText(
            frame,
            f"Brillo promedio: {info['brightness']:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        cv2.putText(
            frame,
            f"Desviacion std: {info['std_dev']:.1f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        cv2.putText(
            frame,
            f"Pixeles oscuros: {info['dark_ratio']*100:.1f}%",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        cv2.putText(
            frame,
            f"Nivel: {info['level']}",
            (10, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        # ---------------------------------------------------------------------
        # Mostramos ventana
        # ---------------------------------------------------------------------
        cv2.imshow("Evaluacion de Iluminacion", frame)

        # ---------------------------------------------------------------------
        # Salida con:
        #
        #   q
        #   ESC
        # ---------------------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break


    # =========================================================================
    # LIBERAR RECURSOS
    # =========================================================================
    cap.release()
    cv2.destroyAllWindows()

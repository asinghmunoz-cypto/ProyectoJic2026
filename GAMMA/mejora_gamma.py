# =============================================================================
# ARCHIVO: mejora_gamma.py
#
# Análisis del problema:
#   La corrección gamma es una de las técnicas de realce más baratas que
#   existen: solo aplica una curva no lineal a cada pixel. Con un gamma menor
#   que 1 la imagen se aclara; con gamma mayor que 1 se oscurece. Es ideal
#   como primer escalón de realce en el sistema adaptativo porque su costo es
#   esencialmente nulo (una sola tabla de 256 entradas) y por lo tanto puede
#   correr a 30+ FPS sin afectar el resto del pipeline.
#   Este script es la prueba aislada de ese bloque. En lugar de usar un gamma
#   fijo, calculamos un gamma "adaptativo" que depende del brillo promedio
#   del frame: cuando la escena está muy oscura usamos un gamma agresivo
#   (alrededor de 0.4) y cuando ya hay buena luz dejamos gamma=1.0 (sin
#   cambio). Así evitamos saturar la imagen en condiciones normales y
#   ahorramos pasar siempre por la corrección, al mismo tiempo que mantenemos
#   un realce automático cuando la luz cae.
#
# Modelo IPO:
#   Entrada : flujo de la cámara web (frames BGR uint8 de OpenCV)
#   Proceso : - por cada frame, medir el brillo promedio en gris
#             - calcular un valor de gamma con gamma_adaptativo
#             - construir una LUT de 256 valores con esa gamma y aplicarla
#               al frame con cv2.LUT
#             - mostrar original y frame corregido lado a lado para
#               compararlos en vivo
#   Salida  : ventana con el par "original | gamma" actualizada en cada
#             iteración, hasta que el usuario presione 'q' o ESC.
# =============================================================================

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# CONSTANTES
#
# Análisis del problema:
#   GAMMA_MIN define qué tan fuerte llega a ser el aclarado en el peor caso
#   (frame casi negro). Bajar GAMMA_MIN aclara más pero amplifica ruido;
#   subirlo deja la imagen más oscura pero más limpia. GAMMA_MAX se fija en
#   1.0 porque en este prototipo no nos interesa oscurecer escenas claras —
#   solo aclarar las oscuras. Mantener GAMMA_MAX=1.0 garantiza que en luz
#   buena la imagen sale idéntica a la original.
# -----------------------------------------------------------------------------
GAMMA_MIN = 0.4   # techo de aclarado para frames muy oscuros
GAMMA_MAX = 1.0   # no oscurecemos nunca


# -----------------------------------------------------------------------------
# FUNCIÓN: gamma_adaptativo
#
# Análisis del problema:
#   Tenemos que mapear "qué tan oscuro está el frame" a "cuánto gamma aplicar".
#   Usamos una interpolación lineal sencilla: cuando el brillo medio es 0
#   (frame negro) devolvemos GAMMA_MIN (aclarado máximo); cuando es 128 o más
#   devolvemos GAMMA_MAX = 1.0 (sin cambio). El cap en 128 evita aclarar
#   imágenes que ya están en rango medio y produce una transición suave entre
#   "oscuro" y "normal". Es lineal a propósito: una curva más compleja
#   tampoco ayudaría sin antes calibrar los umbrales reales en el coche.
#
# Modelo IPO:
#   Entrada : brillo_medio — float en [0, 255], media del canal gris del frame
#   Proceso : - normalizar brillo a [0, 1] con clip al máximo 128/128 = 1
#             - interpolar linealmente entre GAMMA_MIN y GAMMA_MAX usando ese
#               factor t (gamma = GAMMA_MIN + (GAMMA_MAX - GAMMA_MIN) * t)
#   Salida  : float — valor de gamma a usar en aplicar_gamma
# -----------------------------------------------------------------------------
def gamma_adaptativo(brillo_medio):
    t = np.clip(brillo_medio / 128.0, 0.0, 1.0)
    return GAMMA_MIN + (GAMMA_MAX - GAMMA_MIN) * t


# -----------------------------------------------------------------------------
# FUNCIÓN: aplicar_gamma
#
# Análisis del problema:
#   La fórmula de la corrección gamma es out = 255 * (in/255) ** (1/gamma).
#   Si la aplicáramos pixel por pixel con pow(), procesar un frame HD costaría
#   varios milisegundos solo en aritmética en Python. La forma estándar de
#   evitarlo es precalcular una tabla de 256 valores (una para cada nivel de
#   intensidad posible) y usar cv2.LUT, que está implementado en C y resuelve
#   el frame entero en una sola pasada vectorizada. Esto vuelve la operación
#   prácticamente gratis.
#
# Modelo IPO:
#   Entrada : frame — numpy array BGR uint8 de forma (H, W, 3)
#             gamma — float positivo (gamma < 1 aclara, gamma > 1 oscurece)
#   Proceso : - construir una tabla LUT de 256 entradas con la formula
#               directa: tabla[i] = ((i/255) ** gamma) * 255
#             - aplicar la tabla al frame con cv2.LUT
#   Salida  : numpy array BGR uint8 de la misma forma, con gamma corregida
# -----------------------------------------------------------------------------
def aplicar_gamma(frame, gamma):
    # out = in^gamma. Para un pixel oscuro (ej. 0.2):
    #   gamma = 0.4 -> 0.2^0.4 ~= 0.525  (aclara)
    #   gamma = 2.0 -> 0.2^2.0 = 0.04    (oscurece)
    tabla = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(frame, tabla)


# -----------------------------------------------------------------------------
# BUCLE PRINCIPAL
#
# Análisis del problema:
#   Cierra el ciclo lectura → medición → corrección → visualización. Mostramos
#   el frame original a la izquierda y el corregido a la derecha para que la
#   comparación sea inmediata. Los textos overlay incluyen el brillo medio y
#   el valor de gamma usado en ese frame, útiles para entender qué tan
#   agresiva fue la corrección.
#
# Modelo IPO:
#   Entrada : flujo de cv2.VideoCapture(0)
#   Proceso : - leer frame
#             - calcular brillo medio en gris
#             - obtener gamma con gamma_adaptativo y aplicar la corrección
#             - dibujar textos sobre cada mitad del par y concatenar con
#               cv2.hconcat
#             - mostrar la ventana y esperar tecla; salir con 'q' o ESC
#   Salida  : ventana "Original | Gamma adaptativa" actualizada por frame
# -----------------------------------------------------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brillo = float(np.mean(gray))

    gamma = gamma_adaptativo(brillo)
    mejorado = aplicar_gamma(frame, gamma)

    cv2.putText(frame, "Original", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Brillo: {brillo:.1f}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.putText(mejorado, f"Gamma: {gamma:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    grid = cv2.hconcat([frame, mejorado])
    cv2.imshow("Original | Gamma adaptativa", grid)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()

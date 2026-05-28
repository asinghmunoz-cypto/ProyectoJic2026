import time
import cv2


# ============================================
# FUNCIÓN: actualizar_fps
# ============================================
# ENTRADA:
#   fps_anterior — FPS suavizado del frame previo
#   prev_time    — timestamp (time.time()) capturado al inicio del frame previo
#
# PROCESO:
#   1. obtener timestamp actual
#   2. calcular FPS instantáneo = 1 / (ahora - prev_time)
#   3. suavizado exponencial: 0.9 * anterior + 0.1 * instantáneo
#
# SALIDA:
#   tupla (fps_nuevo, ahora) — nuevo FPS suavizado y nuevo prev_time
# ============================================

def actualizar_fps(fps_anterior, prev_time):
    ahora = time.time()
    elapsed = ahora - prev_time
    instantaneo = 1.0 / elapsed if elapsed > 0 else fps_anterior
    fps_nuevo = 0.9 * fps_anterior + 0.1 * instantaneo
    return fps_nuevo, ahora


# ============================================
# DEMO STANDALONE
# Solo corre si ejecutas este archivo directamente:  python fps.py
# Si lo importas, este bloque NO se ejecuta.
# ============================================
if __name__ == "__main__":

    # ============================================
    # variables iniciales
    # ============================================

    cap = cv2.VideoCapture(0)  # 0 = cámara por defecto

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")

    prev_time = time.time()

    fps = 0

    # ============================================
    # loop principal
    # ============================================

    while True:

        # leer frame de la cámara
        ret, frame = cap.read()

        if not ret:
            break

        # ============================================
        # cálculo de FPS suavizado
        # ============================================

        fps, prev_time = actualizar_fps(fps, prev_time)

        # ============================================
        # mostrar FPS en pantalla
        # ============================================

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (30, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        # mostrar frame
        cv2.imshow("FPS Test", frame)

        # salir con tecla q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ============================================
    # liberar recursos
    # ============================================

    cap.release()
    cv2.destroyAllWindows()

import time
import cv2

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
    # aquí va tu algoritmo
    # ============================================

    frame = detect_drowsiness(frame)

    # ============================================
    # cálculo de FPS suavizado
    # ============================================

    current_time = time.time()

    # FPS instantáneo (protección contra división por cero)
    elapsed = current_time - prev_time
    new_fps = 1 / elapsed if elapsed > 0 else fps

    # suavizado del FPS
    fps = (0.9 * fps) + (0.1 * new_fps)

    # actualizar tiempo anterior
    prev_time = current_time

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
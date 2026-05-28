import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

# =====================================================
# FUNCIÓN MAR
# =====================================================

def calculate_mar(mouth):

    # Distancias verticales
    vertical1 = distance.euclidean(mouth[1], mouth[7])
    vertical2 = distance.euclidean(mouth[2], mouth[6])
    vertical3 = distance.euclidean(mouth[3], mouth[5])

    # Distancia horizontal
    horizontal = distance.euclidean(mouth[0], mouth[4])

    # Mouth Aspect Ratio
    mar = (vertical1 + vertical2 + vertical3) / (3.0 * horizontal)

    return mar, vertical1, vertical2, vertical3, horizontal


# =====================================================
# LANDMARKS DE LA BOCA (MediaPipe)
# =====================================================

MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]


# =====================================================
# DEMO STANDALONE
# Solo corre si ejecutas este archivo directamente:  python mar.py
# Si lo importas, este bloque NO se ejecuta.
# =====================================================
if __name__ == "__main__":

    # =====================================================
    # MEDIAPIPE
    # =====================================================

    mp_face_mesh = mp.solutions.face_mesh

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # =====================================================
    # CÁMARA
    # =====================================================

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Reducir tamaño para mejorar FPS
        frame = cv2.resize(frame, (640, 480))

        h, w = frame.shape[:2]

        # Convertir a RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar rostro
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                mouth_points = []

                # Obtener coordenadas de la boca
                for idx in MOUTH:

                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)

                    mouth_points.append((x, y))

                    # Dibujar puntos
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                # Calcular MAR
                mar, v1, v2, v3, h_dist = calculate_mar(mouth_points)

                # Mostrar valores
                cv2.putText(
                    frame,
                    f"MAR: {mar:.2f}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"V1:{v1:.1f} V2:{v2:.1f} V3:{v3:.1f}",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"H:{h_dist:.1f}",
                    (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                # Umbral de bostezo
                if mar > 0.6:

                    cv2.putText(
                        frame,
                        "BOSTEZO DETECTADO",
                        (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )

        cv2.imshow("MAR Test", frame)

        key = cv2.waitKey(1)

        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

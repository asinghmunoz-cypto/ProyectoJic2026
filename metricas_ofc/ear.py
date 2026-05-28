import cv2
import mediapipe as mp
import math

# Índices de los ojos

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Función para calcular distancia entre dos puntos

def distancia(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# Función para calcular EAR

def calcular_EAR(puntos):
    # puntos: lista de 6 coordenadas del ojo

    # verticales
    A = distancia(puntos[1], puntos[5])
    B = distancia(puntos[2], puntos[4])

    # horizontal
    C = distancia(puntos[0], puntos[3])

    # fórmula EAR
    ear = (A + B) / (2.0 * C)
    return ear


# =============================================================================
# DEMO STANDALONE
# Solo corre si ejecutas este archivo directamente:  python ear.py
# Si lo importas, este bloque NO se ejecuta.
# =============================================================================
if __name__ == "__main__":

    # MediaPipe

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
    )

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                left_eye_points = []
                right_eye_points = []

                # Ojo izquierdo
                for idx in LEFT_EYE:
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    left_eye_points.append((x, y))
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                # Ojo derecho
                for idx in RIGHT_EYE:
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    right_eye_points.append((x, y))
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                # Calcular EAR (Eye Aspect Ratio)
                ear_izq = calcular_EAR(left_eye_points)
                ear_der = calcular_EAR(right_eye_points)

                ear_promedio = (ear_izq + ear_der) / 2.0

                # Mostrar en pantalla
                cv2.putText(frame, f"EAR: {ear_promedio:.2f}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("EAR", frame)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break

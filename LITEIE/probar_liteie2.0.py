"""
SCRIPT: probar_liteie.py
cambio de ejemplo 
...
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import cv2
import mediapipe as mp

from metodo_LITEIE import aplicar_liteie

from metricas_ofc.iluminacion import evaluar_iluminacion
from metricas_ofc.ear         import calcular_EAR, LEFT_EYE, RIGHT_EYE
from metricas_ofc.mar         import calculate_mar, MOUTH
from metricas_ofc.psnr        import calcular_psnr
from metricas_ofc.ssim        import calcular_ssim
from metricas_ofc.fps         import actualizar_fps


VIDEO_PATH          = "data/vid2eval.mp4"
PANEL_WIDTH         = 640
APLICAR_SOLO_EN_ROI = True
ROI_PADDING         = 0.10


def _puntos(face_landmarks, indices, w, h):
    """Resolve a list of MediaPipe landmark indices into (x, y) pixel tuples."""
    return [(int(face_landmarks.landmark[i].x * w),
             int(face_landmarks.landmark[i].y * h)) for i in indices]


def medir_ear_mar(frame, face_mesh):
    """
    FUNCTION: medir_ear_mar
    ...
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None, None
    fl = results.multi_face_landmarks[0]
    ear = (calcular_EAR(_puntos(fl, LEFT_EYE,  w, h)) +
           calcular_EAR(_puntos(fl, RIGHT_EYE, w, h))) / 2.0
    mar, *_ = calculate_mar(_puntos(fl, MOUTH, w, h))
    return ear, mar


def bbox_rostro(frame, face_mesh, padding=ROI_PADDING):
    """
    FUNCTION: bbox_rostro
    ...
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    fl = results.multi_face_landmarks[0]
    xs = [lm.x for lm in fl.landmark]
    ys = [lm.y for lm in fl.landmark]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad_x = (x_max - x_min) * padding
    pad_y = (y_max - y_min) * padding
    x0 = max(0, int((x_min - pad_x) * w))
    y0 = max(0, int((y_min - pad_y) * h))
    x1 = min(w, int((x_max + pad_x) * w))
    y1 = min(h, int((y_max + pad_y) * h))
    return x0, y0, x1 - x0, y1 - y0


# CAMBIO: nueva función para dibujar los puntos EAR y MAR sobre un panel,
# igual a como lo hace el código de retinex (círculos verdes para ojos,
# magenta para boca). Devuelve el frame con los puntos ya dibujados.
def _dibujar_landmarks(panel, face_mesh):
    h, w = panel.shape[:2]
    rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return panel
    fl = results.multi_face_landmarks[0]

    # puntos de ojos (verde) y boca (magenta)
    eye_pts   = _puntos(fl, LEFT_EYE,  w, h) + _puntos(fl, RIGHT_EYE, w, h)
    mouth_pts = _puntos(fl, MOUTH, w, h)

    for p in eye_pts:
        cv2.circle(panel, p, 2, (0, 255, 0), -1)
    for p in mouth_pts:
        cv2.circle(panel, p, 2, (255, 0, 255), -1)

    return panel


# CAMBIO: nueva función para dibujar el bounding box de la cara sobre un panel,
# igual a como lo hace el código de retinex (rectángulo azul).
def _dibujar_bbox(panel, face_mesh):
    h, w = panel.shape[:2]
    rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return panel
    fl = results.multi_face_landmarks[0]
    xs = [int(lm.x * w) for lm in fl.landmark]
    ys = [int(lm.y * h) for lm in fl.landmark]
    x0, x1 = max(min(xs), 0), min(max(xs), w)
    y0, y1 = max(min(ys), 0), min(max(ys), h)
    cv2.rectangle(panel, (x0, y0), (x1, y1), (255, 0, 0), 2)
    return panel


def _dibujar(panel, lineas, color):
    for i, texto in enumerate(lineas):
        cv2.putText(panel, texto, (10, 30 + 28 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def _redimensionar(frame, width):
    h, w = frame.shape[:2]
    return cv2.resize(frame, (width, int(h * (width / w))))


face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: could not open video: {VIDEO_PATH}")
    sys.exit(1)

fps = 0.0
prev_time = time.time()

while True:
    ret, frame_original = cap.read()
    if not ret:
        break

    bbox = bbox_rostro(frame_original, face_mesh) if APLICAR_SOLO_EN_ROI else None

    if APLICAR_SOLO_EN_ROI:
        frame_mejorado = frame_original.copy()
        if bbox is not None:
            x, y, ww, hh = bbox
            roi = frame_original[y:y+hh, x:x+ww]
            if roi.size > 0:
                frame_mejorado[y:y+hh, x:x+ww] = aplicar_liteie(roi)
    else:
        frame_mejorado = aplicar_liteie(frame_original)

    if APLICAR_SOLO_EN_ROI and bbox is not None:
        x, y, ww, hh = bbox
        crop_org = frame_original[y:y+hh, x:x+ww]
        crop_mej = frame_mejorado[y:y+hh, x:x+ww]
        info_org = evaluar_iluminacion(crop_org)
        info_mej = evaluar_iluminacion(crop_mej)
        psnr = calcular_psnr(crop_org, crop_mej)
        ssim = calcular_ssim(crop_org, crop_mej)
    elif APLICAR_SOLO_EN_ROI:
        info_org = info_mej = None
        psnr = ssim = None
    else:
        info_org = evaluar_iluminacion(frame_original)
        info_mej = evaluar_iluminacion(frame_mejorado)
        psnr = calcular_psnr(frame_original, frame_mejorado)
        ssim = calcular_ssim(frame_original, frame_mejorado)

    ear_mej, mar_mej = medir_ear_mar(frame_mejorado, face_mesh)
    ear_org, mar_org = medir_ear_mar(frame_original, face_mesh)

    fps, prev_time = actualizar_fps(fps, prev_time)

    panel_izq = _redimensionar(frame_mejorado.copy(), PANEL_WIDTH)
    panel_der = _redimensionar(frame_original.copy(), PANEL_WIDTH)

    # CAMBIO: se dibujan el bbox de la cara y los puntos EAR/MAR
    # en ambos paneles antes de escribir el texto encima,
    # igual a como lo hace el código de retinex.
    panel_izq = _dibujar_bbox(panel_izq, face_mesh)
    panel_der = _dibujar_bbox(panel_der, face_mesh)
    panel_izq = _dibujar_landmarks(panel_izq, face_mesh)
    panel_der = _dibujar_landmarks(panel_der, face_mesh)

    color_mej = (0, 255, 0) if info_mej is not None and info_mej["level"] == "BIEN" else (0, 0, 255)
    color_org = (0, 255, 0) if info_org is not None and info_org["level"] == "BIEN" else (0, 0, 255)

    if psnr is None:
        psnr_txt = "--"
    elif psnr == float("inf"):
        psnr_txt = "inf"
    else:
        psnr_txt = f"{psnr:.2f} dB"
    ssim_txt = "--" if ssim is None else f"{ssim:.4f}"

    lineas_izq = [
        "ENHANCED (LiteIE)",
        f"Illum: {info_mej['brightness']:.1f}  ({info_mej['level']})" if info_mej is not None else "Illum: --",
        f"Dark%: {info_mej['dark_ratio']*100:.1f}%" if info_mej is not None else "Dark%: --",
        f"EAR:   {ear_mej:.3f}" if ear_mej is not None else "EAR:   --",
        f"MAR:   {mar_mej:.3f}" if mar_mej is not None else "MAR:   --",
        f"PSNR:  {psnr_txt}",
        f"SSIM:  {ssim_txt}",
        f"FPS:   {fps:.1f}",
    ]
    lineas_der = [
        "ORIGINAL",
        f"Illum: {info_org['brightness']:.1f}  ({info_org['level']})" if info_org is not None else "Illum: --",
        f"Dark%: {info_org['dark_ratio']*100:.1f}%" if info_org is not None else "Dark%: --",
        f"EAR:   {ear_org:.3f}" if ear_org is not None else "EAR:   --",
        f"MAR:   {mar_org:.3f}" if mar_org is not None else "MAR:   --",
    ]

    _dibujar(panel_izq, lineas_izq, color_mej)
    _dibujar(panel_der, lineas_der, color_org)

    combinado = cv2.hconcat([panel_izq, panel_der])
    cv2.imshow("LiteIE  --  enhanced (left)  vs  original (right)", combinado)

    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        break

cap.release()
cv2.destroyAllWindows()
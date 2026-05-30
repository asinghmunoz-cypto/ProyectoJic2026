"""
SCRIPT: probar_gamma.py

Side-by-side test bench for the adaptive gamma enhancement pipeline.

Reads a video file, runs each frame through aplicar_gamma() (from
metodo_gamma.py), and displays the result alongside the untouched original
so they can be compared visually and quantitatively. All metric functions
are imported from the metricas_ofc/ package — none of them live here.

  LEFT panel : enhanced video with the full metric set
               (illumination, EAR, MAR, PSNR, SSIM, FPS)
  RIGHT panel: original video with only the scene-level metrics
               (illumination, EAR, MAR)

When the run finishes, every per-frame metric is dumped to a CSV and the
averages of the numeric columns are printed to stdout.

To change which video is analysed, edit VIDEO_PATH below.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import csv

import cv2
import mediapipe as mp

from metodo_gamma import aplicar_gamma

from metricas_ofc.iluminacion import evaluar_iluminacion
from metricas_ofc.ear         import calcular_EAR, LEFT_EYE, RIGHT_EYE
from metricas_ofc.mar         import calculate_mar, MOUTH
from metricas_ofc.psnr        import calcular_psnr
from metricas_ofc.ssim        import calcular_ssim
from metricas_ofc.fps         import actualizar_fps


"""
CONSTANTS

VIDEO_PATH:           change this to point at the video file you want to
                      analyse. accepts any format OpenCV can decode.

PANEL_WIDTH:          width (in pixels) of each side-by-side panel. the
                      panel height is derived from the source aspect ratio.

APLICAR_SOLO_EN_ROI:  if True, gamma is only applied inside the face bounding
                      box detected by MediaPipe (the only region the drowsiness
                      metrics actually look at). The rest of the frame stays
                      untouched. If False, gamma is applied to the whole frame.

ROI_PADDING:          extra margin around the face bbox, as a fraction of the
                      bbox size (0.10 = +10%). only used when APLICAR_SOLO_EN_ROI.

CSV_OUTPUT_PATH:      file path where the per-frame metrics are written once the
                      run finishes. resolved relative to this script so the CSV
                      always lands in GAMMA/ no matter where you ran from.
"""
VIDEO_PATH          = "data/vid2eval.mp4"
PANEL_WIDTH         = 640
APLICAR_SOLO_EN_ROI = True
ROI_PADDING         = 0.10
CSV_OUTPUT_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "resultados_gamma.csv")


def _puntos(face_landmarks, indices, w, h):
    """Resolve a list of MediaPipe landmark indices into (x, y) pixel tuples."""
    return [(int(face_landmarks.landmark[i].x * w),
             int(face_landmarks.landmark[i].y * h)) for i in indices]


# Una sola llamada a FaceMesh por frame: devuelve los landmarks y el bbox que
# luego se reutilizan en todo el resto del loop.
def obtener_landmarks(frame, face_mesh):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None, None

    fl = results.multi_face_landmarks[0]

    xs = [int(lm.x * w) for lm in fl.landmark]
    ys = [int(lm.y * h) for lm in fl.landmark]

    x0 = max(min(xs), 0)
    y0 = max(min(ys), 0)
    x1 = min(max(xs), w)
    y1 = min(max(ys), h)

    bbox = (x0, y0, x1 - x0, y1 - y0)

    return fl, bbox


# Calcula EAR y MAR reutilizando los landmarks ya detectados, sin volver a
# llamar a face_mesh.process().
def medir_ear_mar_desde_landmarks(fl, w, h):

    if fl is None:
        return None, None

    ear = (
        calcular_EAR(_puntos(fl, LEFT_EYE, w, h)) +
        calcular_EAR(_puntos(fl, RIGHT_EYE, w, h))
    ) / 2.0

    mar, *_ = calculate_mar(_puntos(fl, MOUTH, w, h))

    return ear, mar


# Dibuja el bounding box escalando las coordenadas del frame original al tamano
# del panel redimensionado, para que el cuadro quede alineado.
def dibujar_bbox(panel, bbox, frame_original_shape):

    if bbox is None:
        return panel

    orig_h, orig_w = frame_original_shape[:2]
    panel_h, panel_w = panel.shape[:2]

    scale_x = panel_w / orig_w
    scale_y = panel_h / orig_h

    x, y, w, h = bbox

    x0 = int(x * scale_x)
    y0 = int(y * scale_y)
    x1 = int((x + w) * scale_x)
    y1 = int((y + h) * scale_y)

    cv2.rectangle(panel, (x0, y0), (x1, y1), (255, 0, 0), 2)

    return panel


# Dibuja los landmarks escalando las coordenadas del frame original al tamano
# del panel redimensionado, para que los puntos queden alineados.
def dibujar_landmarks(panel, fl, frame_original_shape):

    if fl is None:
        return panel

    orig_h, orig_w = frame_original_shape[:2]
    panel_h, panel_w = panel.shape[:2]

    scale_x = panel_w / orig_w
    scale_y = panel_h / orig_h

    eye_pts = (
        _puntos(fl, LEFT_EYE,  orig_w, orig_h) +
        _puntos(fl, RIGHT_EYE, orig_w, orig_h)
    )
    mouth_pts = _puntos(fl, MOUTH, orig_w, orig_h)

    for p in eye_pts:
        cv2.circle(panel, (int(p[0] * scale_x), int(p[1] * scale_y)), 2, (0, 255, 0), -1)

    for p in mouth_pts:
        cv2.circle(panel, (int(p[0] * scale_x), int(p[1] * scale_y)), 2, (255, 0, 255), -1)

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

# Accumulates one dict per processed frame. Dumped to CSV after the loop ends.
metricas_por_frame = []
frame_idx = 0

while True:
    ret, frame_original = cap.read()
    if not ret:
        break

    # Una sola llamada a FaceMesh por frame. fl y bbox se reutilizan en todo
    # el resto del loop.
    fl, bbox = obtener_landmarks(frame_original, face_mesh)

    if APLICAR_SOLO_EN_ROI:
        frame_mejorado = frame_original.copy()
        if bbox is not None:
            x, y, ww, hh = bbox
            roi = frame_original[y:y+hh, x:x+ww]
            if roi.size > 0:
                frame_mejorado[y:y+hh, x:x+ww] = aplicar_gamma(roi)
    else:
        frame_mejorado = aplicar_gamma(frame_original)

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

    # EAR/MAR calculados desde los landmarks ya detectados, sin llamar a
    # FaceMesh de nuevo.
    h, w = frame_original.shape[:2]
    ear_org, mar_org = medir_ear_mar_desde_landmarks(fl, w, h)
    ear_mej, mar_mej = ear_org, mar_org

    fps, prev_time = actualizar_fps(fps, prev_time)

    metricas_por_frame.append({
        "frame":       frame_idx,
        "illum_org":   info_org["brightness"]       if info_org is not None else None,
        "darkpct_org": info_org["dark_ratio"] * 100 if info_org is not None else None,
        "level_org":   info_org["level"]            if info_org is not None else None,
        "ear_org":     ear_org,
        "mar_org":     mar_org,
        "illum_mej":   info_mej["brightness"]       if info_mej is not None else None,
        "darkpct_mej": info_mej["dark_ratio"] * 100 if info_mej is not None else None,
        "level_mej":   info_mej["level"]            if info_mej is not None else None,
        "ear_mej":     ear_mej,
        "mar_mej":     mar_mej,
        "psnr":        psnr,
        "ssim":        ssim,
        "fps":         fps,
    })
    frame_idx += 1

    panel_izq = _redimensionar(frame_mejorado.copy(), PANEL_WIDTH)
    panel_der = _redimensionar(frame_original.copy(), PANEL_WIDTH)

    panel_izq = dibujar_bbox(panel_izq, bbox, frame_original.shape)
    panel_der = dibujar_bbox(panel_der, bbox, frame_original.shape)
    panel_izq = dibujar_landmarks(panel_izq, fl, frame_original.shape)
    panel_der = dibujar_landmarks(panel_der, fl, frame_original.shape)

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
        "ENHANCED (Gamma)",
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
    cv2.imshow("Gamma  --  enhanced (left)  vs  original (right)", combinado)

    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        break

cap.release()
cv2.destroyAllWindows()


# =============================================================================
# AFTER THE RUN: dump per-frame metrics to CSV and print averages on stdout.
# =============================================================================

if metricas_por_frame:

    # ---- CSV dump ---------------------------------------------------------
    with open(CSV_OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metricas_por_frame[0].keys()))
        writer.writeheader()
        writer.writerows(metricas_por_frame)
    print(f"\nCSV saved to: {CSV_OUTPUT_PATH}")

    # ---- Averages of numeric columns --------------------------------------
    # Skip None values (no face detected in ROI mode) and +/-inf values (PSNR
    # of identical frames). Categorical columns (level_org, level_mej) are
    # not averaged — they are not numbers.
    def _promedio(col):
        valores = [r[col] for r in metricas_por_frame
                   if isinstance(r[col], (int, float))
                   and r[col] not in (float("inf"), float("-inf"))]
        return (sum(valores) / len(valores)) if valores else None

    columnas_numericas = [
        "illum_org", "darkpct_org", "ear_org", "mar_org",
        "illum_mej", "darkpct_mej", "ear_mej", "mar_mej",
        "psnr", "ssim", "fps",
    ]

    print(f"\n=== AVERAGES OVER {len(metricas_por_frame)} FRAMES ===")
    for c in columnas_numericas:
        v = _promedio(c)
        print(f"  {c:14s}: {'--' if v is None else f'{v:.4f}'}")
else:
    print("\nNo frames processed — nothing to save.")

"""
SCRIPT: probar_liteie.py

Side-by-side test bench for the LiteIE enhancement pipeline.

Reads a video file, runs each frame through aplicar_liteie() (from
metodo_LITEIE.py), and displays the result alongside the untouched original
so they can be compared visually and quantitatively. All metric functions
are imported from the metricas_ofc/ package — none of them live here.

  LEFT panel : enhanced video with the full metric set
               (illumination, EAR, MAR, PSNR, SSIM, FPS)
  RIGHT panel: original video with only the scene-level metrics
               (illumination, EAR, MAR)

To change which video is analysed, edit VIDEO_PATH below.
"""

import os, sys
# Add LITEIE/ so we can import metodo_LITEIE, and the project root so we can
# import the metricas_ofc package.
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


"""
CONSTANTS

VIDEO_PATH:  change this to point at the video file you want to analyse.
             accepts any format OpenCV can decode (.mp4, .avi, .mov, ...).
             can be absolute or relative to where you run the script from.

PANEL_WIDTH: width (in pixels) of each side-by-side panel. the panel height
             is derived from the source aspect ratio. lower it if your
             monitor cannot fit both panels horizontally.
"""
VIDEO_PATH  = "video.mp4"
PANEL_WIDTH = 640


def _puntos(face_landmarks, indices, w, h):
    """Resolve a list of MediaPipe landmark indices into (x, y) pixel tuples."""
    return [(int(face_landmarks.landmark[i].x * w),
             int(face_landmarks.landmark[i].y * h)) for i in indices]


def medir_ear_mar(frame, face_mesh):
    """
    FUNCTION: medir_ear_mar

    Problem Analysis:
      Runs MediaPipe FaceMesh on a single frame and returns the average
      EAR across both eyes plus the MAR, by calling the imported metric
      functions from metricas_ofc. Returns (None, None) when no face is
      detected so the caller can show a placeholder instead. Detecting
      independently on the original and on the enhanced frame is the
      whole point of this comparison: if the original is too dark,
      FaceMesh may fail there but succeed on the enhanced version.

    IPO Model:
      Input  : frame — BGR uint8 numpy array.
               face_mesh — initialized mp.solutions.face_mesh.FaceMesh.
      Process: - convert BGR -> RGB
               - call face_mesh.process()
               - if a face is found, pull LEFT_EYE / RIGHT_EYE / MOUTH
                 landmark points and call calcular_EAR (twice) and
                 calculate_mar.
      Output : tuple (ear, mar) — both floats, or (None, None) if no face.
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

    frame_mejorado = aplicar_liteie(frame_original)

    info_mej = evaluar_iluminacion(frame_mejorado)
    ear_mej, mar_mej = medir_ear_mar(frame_mejorado, face_mesh)
    psnr = calcular_psnr(frame_original, frame_mejorado)
    ssim = calcular_ssim(frame_original, frame_mejorado)

    info_org = evaluar_iluminacion(frame_original)
    ear_org, mar_org = medir_ear_mar(frame_original, face_mesh)

    fps, prev_time = actualizar_fps(fps, prev_time)

    panel_izq = _redimensionar(frame_mejorado.copy(), PANEL_WIDTH)
    panel_der = _redimensionar(frame_original.copy(), PANEL_WIDTH)

    color_mej = (0, 255, 0) if info_mej["level"] == "BIEN" else (0, 0, 255)
    color_org = (0, 255, 0) if info_org["level"] == "BIEN" else (0, 0, 255)

    psnr_txt = "inf" if psnr == float("inf") else f"{psnr:.2f} dB"
    lineas_izq = [
        "ENHANCED (LiteIE)",
        f"Illum: {info_mej['brightness']:.1f}  ({info_mej['level']})",
        f"Dark%: {info_mej['dark_ratio']*100:.1f}%",
        f"EAR:   {ear_mej:.3f}" if ear_mej is not None else "EAR:   --",
        f"MAR:   {mar_mej:.3f}" if mar_mej is not None else "MAR:   --",
        f"PSNR:  {psnr_txt}",
        f"SSIM:  {ssim:.4f}",
        f"FPS:   {fps:.1f}",
    ]
    lineas_der = [
        "ORIGINAL",
        f"Illum: {info_org['brightness']:.1f}  ({info_org['level']})",
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

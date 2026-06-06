# =============================================================================
# reemplazar_desde_origen.py
#
# Reemplaza automaticamente las imagenes de referencia que NO detectan rostro
# por imagenes NUEVAS tomadas del dataset original (descargado en esta maquina).
#
# Flujo:
#   1. Revisa EN VIVO que imagenes del set de 500 fallan la deteccion de rostro
#      (mismas condiciones que generar_resultados.py: 640x480, static_image_mode).
#   2. Recorre la carpeta --origen (recursivo) y arma candidatas. Por cada mala
#      elige una candidata al azar (semilla fija) y la VALIDA con MediaPipe.
#   3. Copia la buena al set de 500, borra la mala, regenera los 5 niveles
#      oscuros (B5..B40) de las nuevas y limpia los oscuros huerfanos de las
#      que quito.
#   4. Imprime el mapeo quitada -> anadida.
#
# Modos de candidata:
#   - SIN --recortar (por defecto): la candidata es la imagen del origen tal
#     cual (asume que el origen ya son recortes de cara).
#   - CON  --recortar (Opcion A): el origen son FOTOS COMPLETAS. Se detectan los
#     rostros, se recorta cada cara con un margen, se DESCARTAN las que queden
#     mas pequenas que --min-lado (las que luego saldrian pixeladas), y se nombra
#     cada recorte como "<foto>_face_<N>.jpg" para igualar el formato del dataset.
#
# Uso:
#   # En seco (no copia ni borra nada, solo muestra el plan):
#   python reemplazar_desde_origen.py --dataset cerrado --origen "RUTA" --dry-run
#
#   # De verdad, origen ya recortado:
#   python reemplazar_desde_origen.py --dataset cerrado --origen "RUTA"
#
#   # De verdad, origen = fotos completas (recorta caras, descarta pixeladas):
#   python reemplazar_desde_origen.py --dataset cerrado --origen "RUTA" --recortar
#
# Reutiliza las funciones de validar_y_regenerar.py para garantizar criterios
# identicos de deteccion y de oscurecimiento.
# =============================================================================

import os
import sys
import shutil
import random
import argparse

import cv2

RAIZ = os.path.dirname(os.path.abspath(__file__))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

import mediapipe as mp
from validar_y_regenerar import (
    ajustar_brillo, detecta_rostro, DATASETS, NIVELES, EXTS,
)


def listar_imagenes(carpeta):
    return sorted(n for n in os.listdir(carpeta) if n.lower().endswith(EXTS))


def listar_fotos_origen(origen):
    """Rutas de todas las imagenes bajo `origen` (recursivo)."""
    fotos = []
    for raiz, _, archivos in os.walk(origen):
        for a in archivos:
            if a.lower().endswith(EXTS):
                fotos.append(os.path.join(raiz, a))
    return fotos


def recortar_caras(img, nombre_foto, face_det, min_lado, margen):
    """Detecta rostros en una foto completa y devuelve una lista de
    (recorte_bgr, nombre_recorte). Descarta recortes mas pequenos que
    `min_lado`. Nombra como '<nombre_foto>_face_<N>.jpg' (N 1-indexado),
    igualando el formato del dataset."""
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = face_det.process(rgb)
    if not res.detections:
        return []

    recortes = []
    for i, det in enumerate(res.detections, start=1):
        bb = det.location_data.relative_bounding_box
        x = int(bb.xmin * w)
        y = int(bb.ymin * h)
        bw = int(bb.width * w)
        bh = int(bb.height * h)

        # Margen alrededor de la caja para no cortar la cara.
        mx = int(bw * margen)
        my = int(bh * margen)
        x0 = max(0, x - mx)
        y0 = max(0, y - my)
        x1 = min(w, x + bw + mx)
        y1 = min(h, y + bh + my)
        if x1 <= x0 or y1 <= y0:
            continue

        recorte = img[y0:y1, x0:x1]
        rh, rw = recorte.shape[:2]
        if min(rh, rw) < min_lado:        # demasiado pequena -> saldria pixelada
            continue

        nombre = f"{nombre_foto}_face_{i}.jpg"
        recortes.append((recorte, nombre))
    return recortes


def iter_candidatos(args, usados, face_mesh, face_det):
    """Generador de candidatas YA VALIDADAS (detectan rostro) y con nombre no
    usado. Cada item es (img_bgr, nombre). El orden es aleatorio con semilla
    fija. Funciona en modo copia (origen recortado) o modo --recortar."""
    fotos = listar_fotos_origen(args.origen)
    random.seed(args.seed)
    random.shuffle(fotos)

    for ruta in fotos:
        img = cv2.imread(ruta)
        if img is None:
            continue

        if args.recortar:
            base = os.path.basename(ruta)
            for recorte, nombre in recortar_caras(
                    img, base, face_det, args.min_lado, args.margen):
                if nombre in usados:
                    continue
                if detecta_rostro(face_mesh, recorte):
                    yield recorte, nombre
        else:
            nombre = os.path.basename(ruta)
            if nombre in usados:
                continue
            if detecta_rostro(face_mesh, img):
                yield img, nombre


def regenerar_niveles(cfg, nombres, carpeta_ref):
    for nivel in NIVELES:
        carpeta_dest = os.path.join(RAIZ, cfg["dark_tpl"].format(n=nivel))
        os.makedirs(carpeta_dest, exist_ok=True)
        for nombre in nombres:
            img = cv2.imread(os.path.join(carpeta_ref, nombre))
            if img is None:
                continue
            cv2.imwrite(os.path.join(carpeta_dest, nombre),
                        ajustar_brillo(img, nivel))


def borrar_niveles(cfg, nombres):
    """Borra de cada carpeta Bxx los archivos con esos nombres (huerfanos)."""
    borrados = 0
    for nivel in NIVELES:
        carpeta_dest = os.path.join(RAIZ, cfg["dark_tpl"].format(n=nivel))
        if not os.path.isdir(carpeta_dest):
            continue
        for nombre in nombres:
            ruta = os.path.join(carpeta_dest, nombre)
            if os.path.exists(ruta):
                os.remove(ruta)
                borrados += 1
    return borrados


def main():
    parser = argparse.ArgumentParser(
        description="Reemplaza imagenes sin rostro por nuevas del dataset origen.")
    parser.add_argument("--dataset", choices=["cerrado", "abierto"], required=True)
    parser.add_argument("--origen", required=True,
                        help="Carpeta del dataset original (se recorre recursivo).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla para la seleccion aleatoria (default 42).")
    parser.add_argument("--dry-run", action="store_true",
                        help="No copia ni borra nada; solo muestra el plan.")
    # --- Opcion A: recorte de caras desde fotos completas ------------------
    parser.add_argument("--recortar", action="store_true",
                        help="El origen son fotos completas: detecta y recorta "
                             "caras, descartando las que queden pixeladas.")
    parser.add_argument("--min-lado", type=int, default=160,
                        help="Lado minimo (px) de un recorte para aceptarlo "
                             "(solo con --recortar). Default 160.")
    parser.add_argument("--margen", type=float, default=0.2,
                        help="Margen alrededor de la caja de la cara, como "
                             "fraccion (solo con --recortar). Default 0.2.")
    args = parser.parse_args()

    if not os.path.isdir(args.origen):
        print(f"ERROR: no existe la carpeta origen: {args.origen}")
        sys.exit(1)

    cfg = DATASETS[args.dataset]
    carpeta_ref = os.path.join(RAIZ, cfg["ref"])
    if not os.path.isdir(carpeta_ref):
        print(f"ERROR: no existe la carpeta ref: {carpeta_ref}")
        sys.exit(1)

    mp_face_mesh = mp.solutions.face_mesh
    # El detector de cajas solo se necesita en modo --recortar.
    face_det = None

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True,
    ) as face_mesh:

        if args.recortar:
            face_det = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5)

        # --- 1. Detectar las imagenes malas del set actual -----------------
        nombres_set = listar_imagenes(carpeta_ref)
        malas = []
        for nombre in nombres_set:
            img = cv2.imread(os.path.join(carpeta_ref, nombre))
            if img is None or not detecta_rostro(face_mesh, img):
                malas.append(nombre)

        print(f"[{args.dataset}] Imagenes en el set: {len(nombres_set)} | "
              f"sin rostro: {len(malas)}")
        if not malas:
            print("No hay nada que reemplazar.")
            if face_det is not None:
                face_det.close()
            return

        # --- 2 y 3. Emparejar cada mala con una candidata validada ---------
        usados = set(nombres_set)
        gen = iter_candidatos(args, usados, face_mesh, face_det)

        plan = []            # (mala, img_bgr, nombre_nuevo)
        sin_reemplazo = []
        for mala in malas:
            elegida = None
            for img_c, nombre_c in gen:
                if nombre_c in usados:        # carrera defensiva
                    continue
                elegida = (img_c, nombre_c)
                usados.add(nombre_c)
                break
            if elegida:
                plan.append((mala, elegida[0], elegida[1]))
            else:
                sin_reemplazo.append(mala)

        if face_det is not None:
            face_det.close()

    # --- 4. Mostrar plan ----------------------------------------------------
    print("\n" + "=" * 70)
    modo = "RECORTE" if args.recortar else "COPIA"
    print(f"PLAN DE REEMPLAZO ({modo})"
          + ("  (DRY-RUN, no se escribe nada)" if args.dry_run else ""))
    print("=" * 70)
    for mala, _img, nombre_c in plan:
        print(f"   QUITAR  {mala}\n   PONER   {nombre_c}\n")
    if sin_reemplazo:
        print("SIN REEMPLAZO (no se encontro candidata que detecte rostro):")
        for m in sin_reemplazo:
            print(f"   {m}")

    if args.dry_run:
        print("\nDry-run: nada se modifico. Quita --dry-run para aplicar.")
        return

    if not plan:
        print("\nNo se pudo emparejar ninguna. Nada que aplicar.")
        return

    # --- 5. Aplicar: escribir nuevas, borrar malas, regenerar/limpiar oscuros
    nuevas = []
    quitadas = []
    for mala, img_c, nombre_c in plan:
        cv2.imwrite(os.path.join(carpeta_ref, nombre_c), img_c)
        os.remove(os.path.join(carpeta_ref, mala))
        nuevas.append(nombre_c)
        quitadas.append(mala)

    print(f"\nEscritas {len(nuevas)} imagenes nuevas, borradas {len(quitadas)} malas.")

    regenerar_niveles(cfg, nuevas, carpeta_ref)
    print(f"Niveles oscuros B{NIVELES} regenerados para las nuevas.")

    borrados = borrar_niveles(cfg, quitadas)
    print(f"Oscuros huerfanos borrados (de las quitadas): {borrados}")

    print("\nListo. Re-corre generar_resultados.py para cada metodo.")
    print("=" * 70)


if __name__ == "__main__":
    main()

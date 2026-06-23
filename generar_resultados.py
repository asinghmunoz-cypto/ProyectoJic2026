# =============================================================================
# generar_resultados.py
#
# Evalua un metodo de mejora de iluminacion (CLAHE / GAMMA / RETINEX / LiteIE)
# y su efecto sobre la deteccion EAR con MediaPipe. Genera UN CSV por metodo
# con una fila por (imagen x nivel de brillo), cubriendo los dos datasets:
#
#   dataset1/ojoscerrados_500  + ojoscerrados_B{5,10,15,25,40}   -> estado "cerrado"
#   dataset2/Ojos_abiertos500  + ojosabiertos_B{5,10,15,25,40}   -> estado "abierto"
#
# La mejora se aplica AL VUELO sobre la imagen COMPLETA (sin ROI). Esto permite
# medir fps/flops reales del metodo sobre cada imagen.
#
# Uso:
#   python generar_resultados.py --metodo CLAHE
#   python generar_resultados.py --metodo GAMMA
#   python generar_resultados.py --metodo RETINEX
#   python generar_resultados.py --metodo LiteIE
#
# Salida:
#   resultados/resultados_<METODO>.csv
#
# NOTA sobre las metricas reutilizadas:
#   - metricas_ofc/ear.py        -> calcular_EAR(puntos)  + LEFT_EYE/RIGHT_EYE
#   - metricas_ofc/iluminacion.py-> evaluar_iluminacion(frame) -> dict
#                                    {brightness, std_dev, dark_ratio, level}
#   - metricas_ofc/psnr.py       -> calcular_psnr(a, b)
#   - metricas_ofc/ssim.py       -> calcular_ssim(a, b)
#   fps y flops se miden aqui (los modulos fps.py/flops.py no exponen un
#   helper reutilizable que reciba (func, imagen)).
# =============================================================================

import os
import sys
import csv
import time
import math
import argparse

import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# Raiz del proyecto = carpeta donde vive este script. La fijamos en sys.path
# para poder importar metricas_ofc / CLAHE / GAMMA / RETINEX / LITEIE sin
# depender del directorio desde el que se ejecute.
RAIZ = os.path.dirname(os.path.abspath(__file__))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

from metricas_ofc.iluminacion import evaluar_iluminacion
from metricas_ofc.psnr import calcular_psnr
from metricas_ofc.ssim import calcular_ssim
from metricas_ofc.ear import calcular_EAR, LEFT_EYE, RIGHT_EYE


# =============================================================================
# CONFIGURACION
# =============================================================================

NIVELES = ["B5", "B10", "B15", "B25", "B40"]

# Un bloque por dataset.
#   dark_tpl usa {n} = numero del nivel (5, 10, ...).
#   mejorada_tpl usa {metodo} y {nivel} (ej. CLAHE, B5) -> carpeta donde se
#   guardan las imagenes mejoradas si se pasa --guardar-imagenes.
DATASETS = [
    {
        "estado": "cerrado",
        "ref":     "dataset1/ojoscerrados_500",
        "dark_tpl": "dataset1/ojoscerrados_B{n}",
        "mejorada_tpl": "dataset1/ojoscerrados_{metodo}/{nivel}",
    },
    {
        "estado": "abierto",
        "ref":     "dataset2/Ojos_abiertos500",
        "dark_tpl": "dataset2/ojosabiertos_B{n}",
        "mejorada_tpl": "dataset2/ojosabiertos_{metodo}/{nivel}",
    },
]

EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# Tamano estandar SOLO para la deteccion EAR con MediaPipe, manteniendo la
# convencion de los scripts existentes (calcular_ear_cerrados.py usa 640x480).
ANCHO_STD = 640
ALTO_STD = 480

COLUMNAS = [
    "imagen_id", "estado_ojo", "nivel_brillo",
    "brillo_ref", "ear_ref",
    "brillo_oscura", "brillo_mejorada",
    "darkpct_mejorada", "std_mejorada", "nivel_ilum",
    "psnr_vs_ref", "ssim_vs_ref", "psnr_vs_oscura",
    "error_brillo_pct",
    "ear_mejorada", "ear_oscura",
    "ear_error_mej_pct", "ear_error_osc_pct",
    "deteccion_ref", "deteccion_mejorada", "deteccion_ojos",
    "fps", "flops",
]


# =============================================================================
# METODOS DE MEJORA  (import perezoso: solo se carga el que se pide)
# =============================================================================

def cargar_metodo(nombre):
    """Devuelve la funcion aplicar(frame_bgr) -> frame_bgr del metodo pedido."""
    if nombre == "CLAHE":
        from CLAHE.metodo_CLAHE import aplicar_CLAHE
        return aplicar_CLAHE
    if nombre == "GAMMA":
        from GAMMA.metodo_gamma import aplicar_gamma
        return aplicar_gamma
    if nombre == "RETINEX":
        from RETINEX.metodo_RETINEX import aplicar_retinex
        return aplicar_retinex
    if nombre == "LiteIE":
        # metodo_LITEIE.py hace `import model`, asi que LITEIE debe estar en path.
        ruta_liteie = os.path.join(RAIZ, "LITEIE")
        if ruta_liteie not in sys.path:
            sys.path.insert(0, ruta_liteie)
        from LITEIE.metodo_LITEIE import aplicar_liteie
        return aplicar_liteie
    raise ValueError(f"Metodo no valido: {nombre}")


# =============================================================================
# FLOPs
# -----------------------------------------------------------------------------
# Solo LiteIE (red neuronal) tiene FLOPs medibles con thop. Para los metodos
# clasicos (CLAHE/GAMMA/RETINEX) no hay un modelo nn.Module que perfilar, asi
# que se reporta NaN. El valor es constante por resolucion, asi que se cachea.
# =============================================================================

_flops_cache = {}


def medir_flops(metodo_nombre, imagen):
    if metodo_nombre != "LiteIE":
        return float("nan")

    h, w = imagen.shape[:2]
    rw = max(16, (w // 16) * 16)
    rh = max(16, (h // 16) * 16)
    clave = (rh, rw)
    if clave in _flops_cache:
        return _flops_cache[clave]

    try:
        import torch
        from thop import profile
        from LITEIE import metodo_LITEIE as lm

        x = torch.randn(1, 3, rh, rw).to(lm.device)
        macs, _ = profile(lm.litie, inputs=(x,), verbose=False)
        flops = float(macs) * 2.0  # 1 MAC = 2 FLOPs
    except Exception as e:
        print(f"[aviso] No se pudieron medir FLOPs ({e}). Se usara NaN.")
        flops = float("nan")

    _flops_cache[clave] = flops
    return flops


# =============================================================================
# EAR desde imagen con MediaPipe (static_image_mode=True)
# -----------------------------------------------------------------------------
# Devuelve (ear, detectado):
#   - ear = float promedio de ambos ojos, o None si no se detecto cara.
#   - detectado = True/False.
# Se redimensiona a 640x480 para mantener la convencion de los scripts EAR
# existentes (consistencia con el ground truth ya calculado del dataset).
# =============================================================================

def calcular_ear_imagen(face_mesh, imagen_bgr):
    frame = cv2.resize(imagen_bgr, (ANCHO_STD, ALTO_STD))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = face_mesh.process(rgb)

    if not resultados.multi_face_landmarks:
        return None, False

    lm = resultados.multi_face_landmarks[0]

    ojo_izq = [(int(lm.landmark[i].x * ANCHO_STD),
               int(lm.landmark[i].y * ALTO_STD)) for i in LEFT_EYE]
    ojo_der = [(int(lm.landmark[i].x * ANCHO_STD),
               int(lm.landmark[i].y * ALTO_STD)) for i in RIGHT_EYE]

    ear = (calcular_EAR(ojo_izq) + calcular_EAR(ojo_der)) / 2.0
    return float(ear), True


# =============================================================================
# PSNR / SSIM con salvaguarda de tamano
# -----------------------------------------------------------------------------
# Ambas imagenes deben tener el mismo tamano. Si difieren, se redimensiona la
# "movil" (mejorada) al tamano de la referencia antes de calcular.
# =============================================================================

def _igualar_tamano(movil, referencia):
    if movil.shape[:2] != referencia.shape[:2]:
        h, w = referencia.shape[:2]
        movil = cv2.resize(movil, (w, h))
    return movil


def psnr_seguro(movil, referencia):
    return calcular_psnr(_igualar_tamano(movil, referencia), referencia)


def ssim_seguro(movil, referencia):
    return calcular_ssim(_igualar_tamano(movil, referencia), referencia)


# =============================================================================
# PROCESAMIENTO DE UNA IMAGEN x NIVEL
# =============================================================================

def porcentaje_error(valor, referencia):
    """abs(valor - ref)/ref * 100, con NaN si algun operando no es valido."""
    if valor is None or referencia is None:
        return float("nan")
    if isinstance(valor, float) and math.isnan(valor):
        return float("nan")
    if referencia == 0:
        return float("nan")
    return abs(valor - referencia) / referencia * 100.0


def procesar(metodo_nombre, aplicar, face_mesh, estado,
             img_ref, ear_ref, nivel, ruta_dark, nombre, ruta_guardar=None):
    """Devuelve una fila (dict) o None si la imagen oscura no existe/no carga.
    Si `ruta_guardar` no es None, guarda ahi la imagen mejorada."""
    if not os.path.exists(ruta_dark):
        return None
    img_dark = cv2.imread(ruta_dark)
    if img_dark is None:
        return None

    # --- Mejora al vuelo sobre la imagen COMPLETA (sin ROI) + medicion de FPS -
    t0 = time.perf_counter()
    img_mej = aplicar(img_dark)
    dt = time.perf_counter() - t0
    fps = (1.0 / dt) if dt > 0 else float("inf")
    flops = medir_flops(metodo_nombre, img_dark)

    # --- Guardar imagen mejorada (opcional) --------------------------------
    if ruta_guardar is not None:
        os.makedirs(os.path.dirname(ruta_guardar), exist_ok=True)
        cv2.imwrite(ruta_guardar, img_mej)

    # --- Metricas de iluminacion -------------------------------------------
    ilum_ref = evaluar_iluminacion(img_ref)
    ilum_dark = evaluar_iluminacion(img_dark)
    ilum_mej = evaluar_iluminacion(img_mej)

    brillo_ref = ilum_ref["brightness"]
    brillo_oscura = ilum_dark["brightness"]
    brillo_mej = ilum_mej["brightness"]

    # --- Calidad de imagen --------------------------------------------------
    psnr_ref = psnr_seguro(img_mej, img_ref)
    ssim_ref = ssim_seguro(img_mej, img_ref)
    psnr_osc = psnr_seguro(img_mej, img_dark)

    # --- EAR sobre oscura y mejorada ---------------------------------------
    ear_mej, det_mej = calcular_ear_imagen(face_mesh, img_mej)
    ear_osc, _ = calcular_ear_imagen(face_mesh, img_dark)

    ear_mej_val = ear_mej if ear_mej is not None else float("nan")
    ear_osc_val = ear_osc if ear_osc is not None else float("nan")

    return {
        "imagen_id": os.path.splitext(nombre)[0],
        "estado_ojo": estado,
        "nivel_brillo": nivel,
        "brillo_ref": round(brillo_ref, 4),
        "ear_ref": round(ear_ref, 6),
        "brillo_oscura": round(brillo_oscura, 4),
        "brillo_mejorada": round(brillo_mej, 4),
        "darkpct_mejorada": round(ilum_mej["dark_ratio"] * 100.0, 4),
        "std_mejorada": round(ilum_mej["std_dev"], 4),
        "nivel_ilum": ilum_mej["level"],
        "psnr_vs_ref": round(psnr_ref, 4),
        "ssim_vs_ref": round(ssim_ref, 6),
        "psnr_vs_oscura": round(psnr_osc, 4),
        "error_brillo_pct": round(porcentaje_error(brillo_mej, brillo_ref), 4),
        "ear_mejorada": ear_mej_val if math.isnan(ear_mej_val) else round(ear_mej_val, 6),
        "ear_oscura": ear_osc_val if math.isnan(ear_osc_val) else round(ear_osc_val, 6),
        "ear_error_mej_pct": round(porcentaje_error(ear_mej, ear_ref), 4),
        "ear_error_osc_pct": round(porcentaje_error(ear_osc, ear_ref), 4),
        "deteccion_ref": 1,                       # si no, la imagen se salta
        "deteccion_mejorada": 1 if det_mej else 0,   # cara detectada (FaceMesh)
        # ojos ubicados = se pudo calcular el EAR en la mejorada. En el pipeline
        # con FaceMesh equivale a deteccion_mejorada (los landmarks de ojos solo
        # existen si se detecto la cara), pero se reporta aparte por claridad.
        "deteccion_ojos": 0 if math.isnan(ear_mej_val) else 1,
        "fps": round(fps, 3),
        "flops": flops if math.isnan(flops) else round(flops, 1),
    }


# =============================================================================
# RESUMEN POR NIVEL
# =============================================================================

def _media(valores):
    limpios = [v for v in valores if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return (sum(limpios) / len(limpios)) if limpios else float("nan")


def imprimir_resumen(filas, niveles=NIVELES):
    print("\n" + "=" * 92)
    print("RESUMEN POR NIVEL DE BRILLO")
    print("=" * 92)
    print(f"{'Nivel':>6} | {'n':>5} | {'PSNR_ref':>9} | {'SSIM_ref':>9} | "
          f"{'EAR_err_mej%':>12} | {'tasa_det':>9} | {'tasa_det_ojos':>13}")
    print("-" * 92)

    for nivel in niveles:
        sub = [f for f in filas if f["nivel_brillo"] == nivel]
        if not sub:
            continue
        n = len(sub)
        psnr = _media([f["psnr_vs_ref"] for f in sub])
        ssim = _media([f["ssim_vs_ref"] for f in sub])
        ear_err = _media([f["ear_error_mej_pct"] for f in sub])
        tasa = sum(f["deteccion_mejorada"] for f in sub) / n
        tasa_ojos = sum(f["deteccion_ojos"] for f in sub) / n
        print(f"{nivel:>6} | {n:>5} | {psnr:>9.3f} | {ssim:>9.4f} | "
              f"{ear_err:>12.3f} | {tasa:>9.2%} | {tasa_ojos:>13.2%}")
    print("=" * 92)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Genera el CSV de resultados de un metodo de mejora.")
    parser.add_argument("--metodo", required=True,
                        choices=["CLAHE", "GAMMA", "RETINEX", "LiteIE"],
                        help="Metodo de mejora a evaluar.")
    parser.add_argument("--niveles", nargs="+", default=NIVELES,
                        metavar="NIVEL",
                        help="Niveles de brillo a procesar (ej. B5 B10). "
                             "Por defecto todos: " + " ".join(NIVELES) + ".")
    parser.add_argument("--guardar-imagenes", action="store_true",
                        help="Guarda las imagenes mejoradas en "
                             "dataset{1,2}/<estado>_<METODO>/<NIVEL>/.")
    args = parser.parse_args()
    metodo_nombre = args.metodo

    # Normalizar y validar los niveles pedidos (acepta 'b5' o 'B5').
    niveles_sel = []
    for nv in args.niveles:
        nv = nv.upper()
        if not nv.startswith("B"):
            nv = "B" + nv          # admite '5' -> 'B5'
        if nv not in NIVELES:
            parser.error(f"Nivel invalido: {nv}. Validos: {', '.join(NIVELES)}")
        if nv not in niveles_sel:
            niveles_sel.append(nv)
    # Mantener el orden canonico de NIVELES.
    niveles_sel = [n for n in NIVELES if n in niveles_sel]

    print(f"Cargando metodo: {metodo_nombre} ...")
    print(f"Niveles a procesar: {' '.join(niveles_sel)}")
    if args.guardar_imagenes:
        print("Se guardaran las imagenes mejoradas.")
    aplicar = cargar_metodo(metodo_nombre)

    carpeta_salida = os.path.join(RAIZ, "resultados")
    os.makedirs(carpeta_salida, exist_ok=True)
    # Si no se procesan todos los niveles, el nombre del CSV lo refleja para no
    # sobrescribir un run completo (ej. resultados_CLAHE_B5.csv).
    if niveles_sel == NIVELES:
        ruta_csv = os.path.join(carpeta_salida, f"resultados_{metodo_nombre}.csv")
    else:
        sufijo = "_".join(niveles_sel)
        ruta_csv = os.path.join(carpeta_salida,
                                f"resultados_{metodo_nombre}_{sufijo}.csv")

    filas = []

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,   # imprescindible: evita el tracking cacheado
        max_num_faces=1,
        refine_landmarks=True,
    ) as face_mesh:

        for ds in DATASETS:
            estado = ds["estado"]
            carpeta_ref = os.path.join(RAIZ, ds["ref"])

            if not os.path.isdir(carpeta_ref):
                print(f"[aviso] No existe carpeta ref: {carpeta_ref}. Se omite.")
                continue

            nombres = sorted(n for n in os.listdir(carpeta_ref)
                             if n.lower().endswith(EXTS))

            print(f"\nDataset '{estado}': {len(nombres)} imagenes ref, "
                  f"{len(niveles_sel)} niveles -> "
                  f"{len(nombres) * len(niveles_sel)} filas max.")

            barra = tqdm(nombres, desc=f"[{metodo_nombre}/{estado}]", unit="img")
            for nombre in barra:
                ruta_ref = os.path.join(carpeta_ref, nombre)
                img_ref = cv2.imread(ruta_ref)
                if img_ref is None:
                    continue

                # EAR de referencia (ground truth). Si no hay cara -> saltar img.
                ear_ref, det_ref = calcular_ear_imagen(face_mesh, img_ref)
                if not det_ref or ear_ref is None:
                    continue

                for nivel in niveles_sel:
                    n = nivel[1:]  # "B5" -> "5"
                    carpeta_dark = os.path.join(RAIZ, ds["dark_tpl"].format(n=n))
                    ruta_dark = os.path.join(carpeta_dark, nombre)

                    ruta_guardar = None
                    if args.guardar_imagenes:
                        carpeta_mej = os.path.join(
                            RAIZ, ds["mejorada_tpl"].format(
                                metodo=metodo_nombre, nivel=nivel))
                        ruta_guardar = os.path.join(carpeta_mej, nombre)

                    fila = procesar(metodo_nombre, aplicar, face_mesh, estado,
                                    img_ref, ear_ref, nivel, ruta_dark, nombre,
                                    ruta_guardar)
                    if fila is not None:
                        filas.append(fila)

    # --- Guardar CSV --------------------------------------------------------
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNAS)
        writer.writeheader()
        writer.writerows(filas)

    print(f"\nCSV guardado: {ruta_csv}")
    print(f"Filas escritas: {len(filas)}")

    imprimir_resumen(filas, niveles_sel)


if __name__ == "__main__":
    main()

# =============================================================================
# validar_y_regenerar.py
#
# Utilidad para cuando reemplazas imagenes de referencia (las que no detectaban
# rostro) por unas nuevas traidas de la maquina de origen.
#
# Hace dos cosas:
#   1. VALIDA que cada imagen de referencia detecte rostro con MediaPipe, BAJO
#      LAS MISMAS CONDICIONES que generar_resultados.py (resize 640x480,
#      static_image_mode=True, confianza por defecto). Asi, si una imagen pasa
#      aqui, esta garantizado que generar_resultados.py NO la va a saltar.
#   2. REGENERA los 5 niveles de oscurecimiento (B5/B10/B15/B25/B40) SOLO para
#      las imagenes que pasaron la validacion, usando el mismo metodo de
#      oscurecimiento que generar_niveles_iluminacion.py (escala lineal del
#      brillo medio).
#
# Uso tipico (tras dejar las nuevas imagenes en la carpeta 500):
#
#   # Solo revisar que todo detecte rostro, sin tocar nada:
#   python validar_y_regenerar.py --solo-validar
#
#   # Validar y regenerar los niveles oscuros de TODO el dataset cerrado:
#   python validar_y_regenerar.py --dataset cerrado
#
#   # Regenerar SOLO los archivos que reemplazaste (mas rapido):
#   python validar_y_regenerar.py --dataset cerrado --imagenes nueva1.jpg nueva2.jpg
#
#   # Ademas borrar versiones oscuras huerfanas (nombres que ya no estan en 500):
#   python validar_y_regenerar.py --dataset cerrado --limpiar-huerfanos
#
# Reglas:
#   - Una imagen que NO detecta rostro se RECHAZA: no se le generan niveles
#     oscuros y se reporta al final. (Evita volver a meter una imagen mala.)
#   - Los archivos que no son imagen (p.ej. desktop.ini) se ignoran solos.
# =============================================================================

import os
import sys
import argparse

import cv2
import numpy as np
import mediapipe as mp

RAIZ = os.path.dirname(os.path.abspath(__file__))

EXTS = (".jpg", ".jpeg", ".png", ".bmp")
NIVELES = [40, 25, 15, 10, 5]

# Mismas condiciones de deteccion que generar_resultados.py
ANCHO_STD = 640
ALTO_STD = 480

# Config de los dos datasets. dark_tpl usa {n} = numero del nivel.
DATASETS = {
    "cerrado": {
        "ref":      "dataset1/ojoscerrados_500",
        "dark_tpl": "dataset1/ojoscerrados_B{n}",
    },
    "abierto": {
        "ref":      "dataset2/Ojos_abiertos500",
        "dark_tpl": "dataset2/ojosabiertos_B{n}",
    },
}


# =============================================================================
# OSCURECIMIENTO  (identico a generar_niveles_iluminacion.py)
# =============================================================================
def ajustar_brillo(img, brillo_objetivo):
    """Escala linealmente la imagen BGR para que su brillo medio gris sea
    aproximadamente `brillo_objetivo`. Recorta a [0, 255]."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brillo_actual = np.mean(gray)
    if brillo_actual < 1:
        return img
    factor = brillo_objetivo / brillo_actual
    nueva = img.astype(np.float32) * factor
    nueva = np.clip(nueva, 0, 255)
    return nueva.astype(np.uint8)


# =============================================================================
# DETECCION DE ROSTRO  (mismas condiciones que el pipeline)
# =============================================================================
def detecta_rostro(face_mesh, img_bgr):
    frame = cv2.resize(img_bgr, (ANCHO_STD, ALTO_STD))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    return bool(res.multi_face_landmarks)


# =============================================================================
# PROCESO DE UN DATASET
# =============================================================================
def procesar_dataset(nombre_ds, cfg, face_mesh, solo_validar,
                     imagenes_filtro, limpiar_huerfanos):
    carpeta_ref = os.path.join(RAIZ, cfg["ref"])
    if not os.path.isdir(carpeta_ref):
        print(f"[{nombre_ds}] No existe la carpeta ref: {carpeta_ref}. Se omite.")
        return [], []

    nombres = sorted(n for n in os.listdir(carpeta_ref)
                     if n.lower().endswith(EXTS))

    # Si el usuario paso --imagenes, solo trabajamos esos archivos.
    if imagenes_filtro:
        pedidos = set(imagenes_filtro)
        nombres = [n for n in nombres if n in pedidos]
        faltan = pedidos - set(nombres)
        for f in sorted(faltan):
            print(f"[{nombre_ds}] AVISO: '{f}' no esta en {cfg['ref']}.")

    print(f"\n[{nombre_ds}] Validando {len(nombres)} imagenes en {cfg['ref']} ...")

    aceptadas = []
    rechazadas = []

    for nombre in nombres:
        ruta = os.path.join(carpeta_ref, nombre)
        img = cv2.imread(ruta)
        if img is None:
            rechazadas.append((nombre, "no se pudo leer"))
            print(f"   RECHAZADA  {nombre}  (no se pudo leer)")
            continue

        if detecta_rostro(face_mesh, img):
            aceptadas.append(nombre)
        else:
            rechazadas.append((nombre, "sin rostro detectado"))
            print(f"   RECHAZADA  {nombre}  (sin rostro detectado)")

    print(f"[{nombre_ds}] Detectan rostro: {len(aceptadas)} | "
          f"Rechazadas: {len(rechazadas)}")

    # --- Regeneracion de niveles oscuros para las aceptadas ----------------
    if not solo_validar and aceptadas:
        print(f"[{nombre_ds}] Regenerando niveles oscuros de "
              f"{len(aceptadas)} imagenes...")
        for nivel in NIVELES:
            carpeta_dest = os.path.join(RAIZ, cfg["dark_tpl"].format(n=nivel))
            os.makedirs(carpeta_dest, exist_ok=True)
            for nombre in aceptadas:
                img = cv2.imread(os.path.join(carpeta_ref, nombre))
                if img is None:
                    continue
                cv2.imwrite(os.path.join(carpeta_dest, nombre),
                            ajustar_brillo(img, nivel))
        print(f"[{nombre_ds}] Niveles B{NIVELES} regenerados.")

    # --- Limpieza de huerfanos en las carpetas oscuras ---------------------
    if not solo_validar and limpiar_huerfanos:
        validos = set(n for n in os.listdir(carpeta_ref)
                      if n.lower().endswith(EXTS))
        total_borrados = 0
        for nivel in NIVELES:
            carpeta_dest = os.path.join(RAIZ, cfg["dark_tpl"].format(n=nivel))
            if not os.path.isdir(carpeta_dest):
                continue
            for f in os.listdir(carpeta_dest):
                if not f.lower().endswith(EXTS):
                    continue
                if f not in validos:
                    os.remove(os.path.join(carpeta_dest, f))
                    total_borrados += 1
        if total_borrados:
            print(f"[{nombre_ds}] Huerfanos borrados en carpetas oscuras: "
                  f"{total_borrados}")

    return aceptadas, rechazadas


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Valida deteccion de rostro y regenera niveles oscuros.")
    parser.add_argument("--dataset", choices=["cerrado", "abierto", "ambos"],
                        default="ambos",
                        help="Dataset a procesar (default: ambos).")
    parser.add_argument("--solo-validar", action="store_true",
                        help="Solo reporta deteccion; no regenera ni borra nada.")
    parser.add_argument("--imagenes", nargs="*", default=None,
                        help="Nombres de archivo concretos a procesar "
                             "(ej. los reemplazos). Si se omite, procesa todos.")
    parser.add_argument("--limpiar-huerfanos", action="store_true",
                        help="Borra de las carpetas Bxx las imagenes cuyo "
                             "nombre ya no exista en la carpeta 500.")
    args = parser.parse_args()

    objetivos = (["cerrado", "abierto"] if args.dataset == "ambos"
                 else [args.dataset])

    total_aceptadas = 0
    total_rechazadas = []

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
    ) as face_mesh:
        for ds in objetivos:
            aceptadas, rechazadas = procesar_dataset(
                ds, DATASETS[ds], face_mesh,
                args.solo_validar, args.imagenes, args.limpiar_huerfanos)
            total_aceptadas += len(aceptadas)
            total_rechazadas += [(ds, n, motivo) for n, motivo in rechazadas]

    # --- Resumen final ------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"Imagenes que detectan rostro: {total_aceptadas}")
    print(f"Imagenes RECHAZADAS: {len(total_rechazadas)}")
    if total_rechazadas:
        print("\nReemplaza estas (siguen sin detectar rostro):")
        for ds, nombre, motivo in total_rechazadas:
            print(f"   [{ds}] {nombre}  -> {motivo}")
    else:
        print("Todas las imagenes procesadas detectan rostro. ✔")

    if not args.solo_validar:
        print("\nListo. Ya puedes re-correr generar_resultados.py.")
    print("=" * 70)


if __name__ == "__main__":
    main()

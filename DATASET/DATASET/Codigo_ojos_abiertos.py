import os
import random
import shutil

carpeta_origen = r"C:\Users\MovilCity\Downloads\archive\Driver Drowsiness Dataset (DDD)\Non Drowsy"
carpeta_destino = r"C:\Users\MovilCity\Downloads\Ojos_abiertos500"

os.makedirs(carpeta_destino, exist_ok=True)

extensiones = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

imagenes = []

for raiz, _, archivos in os.walk(carpeta_origen):
    for archivo in archivos:
        if archivo.lower().endswith(extensiones):
            imagenes.append(os.path.join(raiz, archivo))

seleccionadas = random.sample(imagenes, 500)

for ruta_imagen in seleccionadas:
    nombre = os.path.basename(ruta_imagen)
    shutil.copy2(
        ruta_imagen,
        os.path.join(carpeta_destino, nombre)
    )

print(f"Se copiaron {len(seleccionadas)} imágenes.")
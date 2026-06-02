import os
import random
import shutil
import os

print("Estoy ejecutándome desde:")
print(os.getcwd())
# ==================================================
# CONFIGURACIÓN
# ==================================================

ORIGEN = r"C:\Users\asing\Downloads\dataset_B_FacialImages_highResolution\dataset_B_FacialImages_highResolution"

DESTINO = os.path.join(
    "dataset1",
    "ojoscerrados_500"
)
print("Destino:")
print(os.path.abspath(DESTINO))
N_MUESTRAS = 500

SEED = 42

# ==================================================
# CREAR CARPETA DESTINO
# ==================================================

os.makedirs(DESTINO, exist_ok=True)

# ==================================================
# LISTAR IMÁGENES
# ==================================================

imagenes = [
    f for f in os.listdir(ORIGEN)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp")
    )
]

print(f"Imágenes encontradas: {len(imagenes)}")

# ==================================================
# SELECCIÓN ALEATORIA FIJA
# ==================================================

random.seed(SEED)

seleccionadas = random.sample(
    imagenes,
    N_MUESTRAS
)

# ==================================================
# COPIAR
# ==================================================

for img in seleccionadas:

    shutil.copy(
        os.path.join(ORIGEN, img),
        os.path.join(DESTINO, img)
    )

print(f"Copiadas {len(seleccionadas)} imágenes")

# ==================================================
# GUARDAR LISTA
# ==================================================

with open(
    os.path.join(
        "..",
        "dataset1",
        "muestra_cerrados.txt"
    ),
    "w"
) as f:

    for img in seleccionadas:
        f.write(img + "\n")

print("Lista guardada")
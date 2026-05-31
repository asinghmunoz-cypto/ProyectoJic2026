import cv2

# ==================================================
# Parámetros CLAHE
# ==================================================

CLAHE_CLIP_LIMIT = 4.0
CLAHE_TILE_GRID = (8, 8)

clahe = cv2.createCLAHE(
    clipLimit=CLAHE_CLIP_LIMIT,
    tileGridSize=CLAHE_TILE_GRID
)

def aplicar_CLAHE(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_mejorado = clahe.apply(l)
    lab_mejorado = cv2.merge((l_mejorado, a, b))
    frame_mejorado = cv2.cvtColor(
        lab_mejorado,
        cv2.COLOR_LAB2BGR
    )
    return frame_mejorado

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np


"""
CONSTANTS

GAMMA_MIN: techo de aclarado para el peor caso (frame casi negro). Bajarlo
           aclara mas pero amplifica ruido; subirlo deja la imagen mas oscura
           pero mas limpia.

GAMMA_MAX: en este prototipo no oscurecemos nunca, asi que se fija en 1.0 para
           que en luz buena el frame salga identico al original.
"""
GAMMA_MIN = 0.4
GAMMA_MAX = 1.0


def gamma_adaptativo(brillo_medio):
    """
    FUNCTION: gamma_adaptativo

    Problem Analysis:
      Maps "how dark the frame is" to "how much gamma to apply" with a simple
      linear interpolation: a brightness of 0 (black frame) returns GAMMA_MIN
      (maximum brightening) and a brightness of 128 or more returns GAMMA_MAX =
      1.0 (no change). Capping at 128 avoids brightening frames that are already
      in mid-range and produces a smooth transition between "dark" and "normal".

    IPO Model:
      Input  : brillo_medio — float in [0, 255], mean of the frame's gray channel.
      Process: - normalize brightness to [0, 1] clipping at 128/128 = 1
               - linearly interpolate between GAMMA_MIN and GAMMA_MAX with that
                 factor t (gamma = GAMMA_MIN + (GAMMA_MAX - GAMMA_MIN) * t)
      Output : float — the gamma value to feed aplicar_gamma.
    """
    t = np.clip(brillo_medio / 128.0, 0.0, 1.0)
    return GAMMA_MIN + (GAMMA_MAX - GAMMA_MIN) * t


def aplicar_gamma(frame):
    """
    FUNCTION: aplicar_gamma

    Problem Analysis:
      Enhances a low-light frame with an adaptive gamma correction. Gamma is one
      of the cheapest enhancement techniques: it just applies a non-linear curve
      to every pixel via a 256-entry lookup table, so it runs at 30+ FPS without
      touching the rest of the pipeline. Instead of a fixed gamma we measure the
      frame's average brightness and pick an adaptive gamma (aggressive ~0.4 when
      dark, 1.0 when already lit) so we never over-saturate well-lit scenes. The
      gamma formula out = 255 * (in/255) ** gamma is precomputed into a LUT and
      applied whole-frame with cv2.LUT (implemented in C, one vectorized pass).
      The single-argument signature matches aplicar_liteie() so probar_gamma can
      call it exactly the same way LiteIE is called.

    IPO Model:
      Input  : frame — numpy array of shape (H, W, 3), BGR uint8, the format
                       OpenCV uses when reading a camera or video file.
      Process: - measure mean brightness on the gray channel
               - obtain the adaptive gamma with gamma_adaptativo
               - build a 256-entry LUT: tabla[i] = ((i/255) ** gamma) * 255
               - apply the LUT to the frame with cv2.LUT
      Output : numpy array of shape (H, W, 3), BGR uint8 — the enhanced frame,
               same shape and dtype as the input.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brillo = float(np.mean(gray))
    gamma = gamma_adaptativo(brillo)
    tabla = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(frame, tabla)

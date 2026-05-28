"""
SCRIPT: metodo_retinex.py

Holds the function used to apply Single-Scale Retinex (SSR) enhancement to
a video frame. Lives next to probar_retinex.py so it can be imported
without dragging any test / camera / loop side-effects, mirroring the
metodo_LITEIE.py / probar_liteie.py pair.
"""

import cv2
import numpy as np


"""
CONSTANTS

GAUSSIAN_SIGMA:          standard deviation of the Gaussian used to estimate
                         the illumination layer. larger values give a
                         smoother illumination map (more aggressive
                         enhancement of dark regions); smaller values keep
                         more local detail in the illumination estimate.

ILLUM_FLOOR:             clamp value applied to the illumination estimate
                         before dividing by it. prevents divide-by-zero on
                         near-black pixels which would otherwise produce inf
                         values in the reflectance map.

BILATERAL_D:             kernel diameter for the bilateral smoothing of the
                         reflectance layer. 5 is a small kernel that removes
                         noise without softening edges.

BILATERAL_SIGMA_COLOR:   color sigma for the bilateral filter. larger means
                         more different colors get averaged together.

BILATERAL_SIGMA_SPACE:   spatial sigma for the bilateral filter. larger
                         means pixels further apart influence each other.

GAIN:                    brightness multiplier applied after normalization.
                         1.2 nudges the final image up by 20%, then clips
                         to [0, 1].
"""
GAUSSIAN_SIGMA        = 15
ILLUM_FLOOR           = 0.01
BILATERAL_D           = 5
BILATERAL_SIGMA_COLOR = 50
BILATERAL_SIGMA_SPACE = 50
GAIN                  = 1.2


def aplicar_retinex(frame):
    """
    FUNCTION: aplicar_retinex

    Problem Analysis:
      Applies Single-Scale Retinex (SSR) to a low-light frame. The Retinex
      model assumes the observed image is the product of an illumination
      layer (slowly-varying lighting in the scene) and a reflectance layer
      (the actual surface colors and textures). By estimating the
      illumination with a large Gaussian blur, dividing it out to recover
      the reflectance, smoothing the reflectance to suppress noise, and
      recombining with a final brightness boost, we end up with an image
      where dark areas have been lifted without washing out the bright
      regions. Pure classical CV — no neural network, no weights to load.

    IPO Model:
      Input  : frame — numpy array (H, W, 3), BGR uint8 (OpenCV format).
      Process: - cast to float32 in [0, 1]
               - estimate illumination via large-sigma Gaussian blur
               - clamp illumination to ILLUM_FLOOR to avoid divide-by-zero
               - reflectance = image / illumination
               - bilateral-filter the reflectance to clean up noise
               - recombine: enhanced = illumination * reflectance
               - min/max normalize to [0, 1]
               - multiply by GAIN and clip back to [0, 1]
               - scale to [0, 255] and cast to uint8
      Output : numpy array (H, W, 3), BGR uint8 — enhanced frame, same
               shape and dtype as the input.
    """
    img = frame.astype(np.float32) / 255.0

    illumination = cv2.GaussianBlur(img, (0, 0), sigmaX=GAUSSIAN_SIGMA)
    illumination = np.maximum(illumination, ILLUM_FLOOR)

    reflectance = img / illumination
    reflectance = cv2.bilateralFilter(
        reflectance.astype(np.float32),
        d=BILATERAL_D,
        sigmaColor=BILATERAL_SIGMA_COLOR,
        sigmaSpace=BILATERAL_SIGMA_SPACE,
    )

    enhanced = illumination * reflectance
    enhanced = cv2.normalize(enhanced, None, 0, 1, cv2.NORM_MINMAX)
    enhanced = np.clip(enhanced * GAIN, 0, 1)

    return (enhanced * 255).astype(np.uint8)

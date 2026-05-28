# =============================================================================
# FILE: model.py
#
# Problem Analysis:
#   This file defines LiteIE, a lightweight neural network for low-light image
#   enhancement. The problem it solves is: given a dark/underexposed photo,
#   produce a brighter, more visually correct version — without needing paired
#   training data at inference time.
#   The design philosophy is extreme efficiency: instead of a large encoder-
#   decoder like most image restoration networks, LiteIE uses a single tiny
#   convolutional block (58 parameters total) shared across multiple passes.
#   It learns a per-pixel "curve" that nudges each pixel up or down, then
#   applies that curve repeatedly. The number of repetitions adapts to how
#   dark the image is, so very dark images get more correction passes.
#
# IPO Model:
#   Input  : a low-light RGB image as a float32 tensor, shape (1, 3, H, W),
#            pixel values normalised to [0.0, 1.0]
#   Process: - run the image through shared_conv three times to learn a
#              per-pixel correction map called "curves"
#            - measure average brightness to decide how many iterations to run
#            - repeatedly apply the curve adjustment formula to the image
#   Output : enhanced tensor of the same shape (1, 3, H, W), brighter pixels
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -----------------------------------------------------------------------------
# VARIABLE: coeffs
#
# Problem Analysis:
#   Very dark images need more enhancement passes than bright ones. This list
#   holds the five coefficients of a 4th-degree polynomial that maps average
#   image brightness (a number in [0,1]) to a number of iterations (1–15).
#   The polynomial was fitted so the curve decreases as brightness increases:
#   near-black images get ~15 iterations; near-average images get ~1.
#   Using a polynomial instead of a lookup table gives a smooth, continuous
#   mapping between any brightness value and its iteration count.
#
# IPO Model:
#   Input  : image mean brightness value in [0.0, 1.0]
#   Process: - evaluate  c0 + c1·v + c2·v² + c3·v³ + c4·v⁴
#            - round to nearest integer and clip to [1, 15]
#            - special case: if brightness < 0.1, always use 8 iterations
#   Output : integer number of enhancement iterations to apply
# -----------------------------------------------------------------------------
coeffs = [10.6301, -49.3563, 193.9553, -507.2211, 582.7439]


class LiteIE(nn.Module):

    # -------------------------------------------------------------------------
    # METHOD: __init__
    #
    # Problem Analysis:
    #   Defines the architecture of the model — what layers it has and how they
    #   are wired together. The key design choice here is the bottleneck: the
    #   first conv compresses 3 RGB channels down to 1, and the second expands
    #   back to 3. This bottleneck (1 channel) is what keeps the total parameter
    #   count at just 58. The same shared_conv block is reused three times in
    #   forward(), so there are no separate layer copies — weight sharing reduces
    #   the model size further while still allowing multi-step feature extraction.
    #
    # IPO Model:
    #   Input  : none (called automatically when you write LiteIE())
    #   Process: - register a LeakyReLU activation (slope 0.01 for negative values)
    #            - build shared_conv as a Sequential of two Conv2d layers:
    #                Conv2d(3 → 1, kernel 3×3, padding 1) — compresses to bottleneck
    #                Conv2d(1 → 3, kernel 3×3, padding 1) — expands back to RGB
    #   Output : an initialised LiteIE object ready to be trained or loaded
    # -------------------------------------------------------------------------
    def __init__(self):
        super(LiteIE, self).__init__()

        self.relu = nn.LeakyReLU(0.01)

        bottleneck_channels = 1
        self.shared_conv = nn.Sequential(
            nn.Conv2d(3, bottleneck_channels, 3, padding=1, bias=True),
            nn.Conv2d(bottleneck_channels, 3, 3, padding=1, bias=True)
        )

    # -------------------------------------------------------------------------
    # METHOD: forward
    #
    # Problem Analysis:
    #   This is the enhancement algorithm itself. It runs in two stages:
    #
    #   Stage 1 — Curve learning:
    #     shared_conv is applied three times in sequence. Each pass refines the
    #     feature representation: x1 captures low-level edges/gradients, x2
    #     builds on that with a second pass, and finally "curves" is produced
    #     with tanh to squash values into [-1, 1]. The curves tensor is a
    #     per-pixel, per-channel correction map — it encodes how much and in
    #     which direction each pixel should be shifted.
    #
    #   Stage 2 — Iterative adjustment:
    #     The formula  x = x + curves * (x² - x)  is applied N times.
    #     For a pixel value x in [0,1]: x² - x = x(x-1) is always ≤ 0.
    #     So the sign of curves controls the direction of change:
    #       - negative curves → subtracts a negative number → pixel gets brighter
    #       - positive curves → subtracts a positive number → pixel gets darker
    #     The model learns curves to be negative where the image is too dark,
    #     causing those pixels to brighten with each iteration. Running the loop
    #     multiple times stacks the effect progressively.
    #     N is computed from the polynomial in coeffs: darker images get more
    #     iterations, up to a maximum of 15.
    #
    # IPO Model:
    #   Input  : x — float32 tensor of shape (1, 3, H, W), values in [0.0, 1.0]
    #   Process: - x1    = ReLU(shared_conv(x))        — first feature pass
    #            - x2    = ReLU(shared_conv(x1))        — second feature pass
    #            - curves = tanh(shared_conv(x2))        — correction map in [-1,1]
    #            - value = mean pixel brightness of x (scalar)
    #            - N = 8 if value < 0.1, else polynomial(value) clipped to [1,15]
    #            - repeat N times: x = x + curves * (x² - x)
    #   Output : enhanced tensor of shape (1, 3, H, W), same dtype as input
    # -------------------------------------------------------------------------
    def forward(self, x):

        x1 = self.relu(self.shared_conv(x))
        x2 = self.relu(self.shared_conv(x1))
        curves = torch.tanh((self.shared_conv(x2)))

        value = x.mean().item()

        for _ in range(8 if value < 0.1 else int(np.clip(np.round(coeffs[0] + coeffs[1]*value + coeffs[2]*value**2 + coeffs[3]*value**3 + coeffs[4]*value**4), 1, 15))):
            x = x + curves * (x*x - x)
        return x

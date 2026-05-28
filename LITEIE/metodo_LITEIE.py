import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import model


"""
CONSTANTS

MODEL_PATH: path to the trained LiteIE checkpoint (.pth file).
            resolved relative to this file's location (LITEIE/) so it works
            no matter what directory you run from.
            replace this if you train a new model or move the weights.
"""
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "snapshots", "epoth_199.pth")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
litie = model.LiteIE().to(device)
litie.load_state_dict(torch.load(MODEL_PATH, map_location=device))
litie.eval()


def aplicar_liteie(frame):
    """
    FUNCTION: aplicar_liteie

    Problem Analysis:
      Enhances a low-light frame using the LiteIE neural network. LiteIE is a
      lightweight image-enhancement model that learns to brighten dark regions
      and recover detail without blowing out the already-lit areas. The frame
      is resized so its height and width are multiples of 16 (the model needs
      that to avoid shape mismatches in its downsampling path), passed through
      the network, and then resized back to the original resolution so the
      output drops in as a replacement for the input frame.

    IPO Model:
      Input  : frame — numpy array of shape (H, W, 3), BGR uint8, the format
                       OpenCV uses when reading a camera or video file.
      Process: - convert BGR -> RGB (the model was trained on RGB)
               - resize to (rw, rh), nearest multiples of 16 >= 16
               - normalize to [0, 1] and convert to a torch tensor of shape
                 (1, 3, rh, rw) on the active device (GPU if available)
               - run the LiteIE forward pass in inference mode
               - bilinear-resize the result back to (H, W) if it changed
               - clamp to [0, 1], scale to [0, 255], cast to uint8
               - convert RGB -> BGR so the caller gets the same format it
                 gave us
      Output : numpy array of shape (H, W, 3), BGR uint8 — the enhanced frame,
               same shape and dtype as the input.
    """
    h, w = frame.shape[:2]
    rw = max(16, (w // 16) * 16)
    rh = max(16, (h // 16) * 16)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (rw, rh))
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device, non_blocking=True)
    with torch.inference_mode():
        enhanced = litie(tensor)
    if (rw, rh) != (w, h):
        enhanced = F.interpolate(enhanced, size=(h, w), mode="bilinear", align_corners=False)
    out = enhanced.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    out = (out * 255.0).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

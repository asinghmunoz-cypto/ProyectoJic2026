import cv2
import torch
import numpy as np
from torchvision import transforms
from model import enhance_net_nopool

# =====================================================
# DEVICE
# =====================================================

device = torch.device("cpu")

# =====================================================
# LOAD MODEL
# =====================================================

DCE_net = enhance_net_nopool().to(device)

DCE_net.load_state_dict(
    torch.load(
        "models/Epoch99.pth",
        map_location=device
    )
)

DCE_net.eval()

# =====================================================
# TRANSFORM
# =====================================================

transform = transforms.ToTensor()

# =====================================================
# CAMERA
# =====================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# =====================================================
# MAIN LOOP
# =====================================================
frame_count = 0
last_enhanced = None

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (320, 240))

    # =================================================
    # PREPROCESS
    # =================================================

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0).to(device)

    # =================================================
    # ZERO-DCE (solo cada 3 frames)
    # =================================================

    frame_count += 1

    if frame_count % 3 == 0:

        with torch.no_grad():
            _, enhanced_image, _ = DCE_net(img)

        enhanced_image = enhanced_image.squeeze(0)
        enhanced_image = enhanced_image.cpu().numpy()
        enhanced_image = np.transpose(
            enhanced_image,
            (1, 2, 0)
        )
        enhanced_image = np.clip(
            enhanced_image * 255.0,
            0,
            255
        ).astype(np.uint8)
        enhanced_image = cv2.cvtColor(
            enhanced_image,
            cv2.COLOR_RGB2BGR
        )

        last_enhanced = enhanced_image

    # =================================================
    # SHOW (reutilizar último frame procesado)
    # =================================================

    if last_enhanced is not None:

        cv2.imshow(
            "Zero-DCE",
            last_enhanced
        )

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

# =====================================================
# END
# =====================================================

cap.release()

cv2.destroyAllWindows()

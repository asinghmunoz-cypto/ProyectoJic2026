import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# DEMO STANDALONE
# Solo corre si ejecutas este archivo directamente:  python flops.py
# Si lo importas, este bloque NO se ejecuta (no hay funciones reusables aquí,
# este script es una medición one-shot del modelo).
# =============================================================================
if __name__ == "__main__":

    import torch
    from thop import profile  # thop hooks into PyTorch to count ops as the model runs
    import model

    # Change these to match the resolution of your actual test images.
    # FLOPs scale linearly with resolution: double the pixels = double the FLOPs.
    WIDTH  = 600
    HEIGHT = 400

    # .eval() disables dropout/batchnorm training behaviour (doesn't affect LiteIE
    # here, but good practice before any profiling or inference).
    net = model.LiteIE().eval()

    # Dummy input: batch of 1, RGB (3 channels), HEIGHT x WIDTH.
    # thop runs a real forward pass with this tensor to count operations.
    # Values don't matter — shape is what drives the FLOPs count.
    x = torch.randn(1, 3, HEIGHT, WIDTH)

    # profile() returns:
    #   macs    — Multiply-Accumulate operations (1 multiply + 1 add = 1 MAC)
    #   params  — total number of learnable parameters in the model
    macs, params = profile(net, inputs=(x,), verbose=False)

    # FLOPs = MACs x2 because each MAC counts as 2 floating-point operations.
    # Many papers report MACs; many tools report FLOPs — this converts between them.
    flops = macs * 2

    # Each parameter is stored as a 32-bit float = 4 bytes.
    # This tells you the minimum RAM needed to hold the model weights.
    param_bytes = params * 4

    print(f"Input resolution : {WIDTH} x {HEIGHT}")
    print(f"Parameters       : {int(params):,}")
    print(f"Model size       : {param_bytes / 1024:.2f} KB  (float32)")
    print(f"MACs             : {macs / 1e6:.3f} M")
    print(f"FLOPs            : {flops / 1e6:.3f} M  (MACs x2)")
    print()
    # thop only counts operations inside nn.Module layers (the two Conv2d layers).
    # The iterative loop in LiteIE.forward() is plain Python, so thop misses it.
    # Its cost is small (element-wise ops) but grows with the number of iterations (1-15).
    print("Note: the dynamic enhancement loop is not captured by thop.")
    print("Real FLOPs at runtime are slightly higher depending on image brightness.")

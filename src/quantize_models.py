import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import numpy as np

def benchmark_and_quantize():
    print("=" * 60)
    print("UNLET-ADAS: Zero-Budget INT8 Quantization Benchmark")
    print("=" * 60)
    
    from src.model import build_model
    model = build_model().eval()
    weights = r"D:\UNLET-ADAS\app\zerodce_cbam_best.pt"
    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights, map_location='cpu'))
        print(f"Loaded weights from {weights}")
    
    # Standard 256x256 proxy tensor
    x = torch.randn(4, 3, 256, 256)
    
    # 1. Benchmark FP32 Baseline
    # Warmup
    for _ in range(5):
        _ = model(x)
        
    t0 = time.time()
    n_iters = 30
    for _ in range(n_iters):
        with torch.no_grad():
            _ = model(x)
    t_fp32 = (time.time() - t0) / (n_iters * 4) * 1000.0
    print(f"FP32 Baseline Latency : {t_fp32:.2f} ms / frame")
    
    # 2. Dynamic INT8 Quantization (CPU AVX-512 / VNNI)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            q_model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Conv2d}, dtype=torch.qint8
            )
        
        # Warmup
        for _ in range(5):
            _ = q_model(x)
            
        t0 = time.time()
        for _ in range(n_iters):
            with torch.no_grad():
                _ = q_model(x)
        t_int8 = (time.time() - t0) / (n_iters * 4) * 1000.0
        speedup = t_fp32 / max(t_int8, 1e-4)
        print(f"INT8 Quantized Latency: {t_int8:.2f} ms / frame ({speedup:.2f}x speedup)")
        
        # Save quantized weights
        q_weights_path = r"D:\UNLET-ADAS\app\zerodce_cbam_int8.pt"
        torch.save(q_model.state_dict(), q_weights_path)
        print(f"Saved INT8 Model      : {q_weights_path}")
    except Exception as e:
        print(f"INT8 dynamic quantization skipped ({e})")
        
    print("=" * 60)

if __name__ == '__main__':
    benchmark_and_quantize()

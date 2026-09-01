"""
Temporary diagnostic — NOT a real test. Confirms the fix in
src/depth.py's load_midas() (pre-trusting the nested
rwightman/gen-efficientnet-pytorch repo that MiDaS_small's backbone
loads internally) actually resolves the EOFError seen in CI before
this fix. Delete once confirmed working.
"""
import torch


def test_midas_load_diagnostic():
    from src.depth import load_midas

    device = torch.device('cpu')
    model, transform, ok = load_midas(device)
    print(f'[midas diag] load_midas() -> ok={ok}, '
          f'model={type(model).__name__ if model else None}')
    assert ok, 'load_midas() returned ok=False even after the trust-list fix'

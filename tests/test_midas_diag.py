"""
Temporary diagnostic — NOT a real test. src/depth.py's load_midas()
deliberately swallows every exception so the app degrades gracefully,
which also means CI's "[model status] MiDaS depth model unavailable"
line doesn't say *why*. This calls torch.hub directly, unwrapped, to
print the actual traceback into the CI log. Delete once the real
cause is known.
"""
import traceback


def test_midas_load_diagnostic():
    import torch
    try:
        model = torch.hub.load(
            'intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
        print('[midas diag] load_state_dict SUCCESS:', type(model))
    except Exception:
        print('[midas diag] FAILED:')
        traceback.print_exc()

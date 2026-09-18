"""
Adds a --no_cbam ablation switch to UNLET-ADAS.

Run once from the repository root:

    python add_cbam_ablation.py

It edits four files in place:
  src/model.py            - ZeroDCECBAM/build_model take use_cbam=True
  src/train.py            - new --no_cbam flag
  src/evaluate.py         - auto-detects CBAM from the checkpoint
  src/measure_sharpness.py- same auto-detection

With use_cbam=False the seven CBAM modules become nn.Identity, so the
network drops from 21,769 to 19,291 parameters and the checkpoint simply
contains no cb*.* keys. The loaders detect that and build the matching
architecture, so no flag is needed at evaluation time.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def patch(rel, edits):
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        print(f'SKIP  {rel} (not found)')
        return
    src = open(path, encoding='utf-8').read()
    done = 0
    for old, new in edits:
        if new in src:                      # already applied
            done += 1
            continue
        if old not in src:
            print(f'WARN  {rel}: pattern not found:\n      {old[:70]}')
            continue
        src = src.replace(old, new, 1)
        done += 1
    open(path, 'w', encoding='utf-8').write(src)
    print(f'OK    {rel}  ({done}/{len(edits)} edits)')


# ---------------------------------------------------------------- model.py
model_edits = [
    (
        "    def __init__(self, num_iters=8, channels=32):\n"
        "        super().__init__()\n"
        "        self.num_iters = num_iters\n",

        "    def __init__(self, num_iters=8, channels=32, use_cbam=True):\n"
        "        super().__init__()\n"
        "        self.num_iters = num_iters\n"
        "        self.use_cbam = use_cbam\n"
        "        # Ablation: with use_cbam=False every attention module\n"
        "        # becomes an identity, removing 2,478 parameters.\n"
        "        def attn(c):\n"
        "            return CBAM(c) if use_cbam else nn.Identity()\n",
    ),
    (
        "def build_model(num_iters=8, channels=32):\n"
        "    \"\"\"Build and return the UNLET-ADAS enhancement model.\"\"\"\n"
        "    model  = ZeroDCECBAM(num_iters=num_iters, channels=channels)\n",

        "def build_model(num_iters=8, channels=32, use_cbam=True):\n"
        "    \"\"\"Build and return the UNLET-ADAS enhancement model.\"\"\"\n"
        "    model  = ZeroDCECBAM(num_iters=num_iters, channels=channels,\n"
        "                         use_cbam=use_cbam)\n",
    ),
]
for n in range(1, 8):
    model_edits.append((f'self.cb{n} = CBAM(channels)',
                        f'self.cb{n} = attn(channels)'))
model_edits.append((
    "    print(f'Iterations : {num_iters}')",
    "    print(f'Iterations : {num_iters}')\n"
    "    print(f'CBAM       : {\"on\" if use_cbam else \"OFF (ablation)\"}')",
))
patch('src/model.py', model_edits)

# ---------------------------------------------------------------- train.py
patch('src/train.py', [
    (
        "    model     = build_model().to(DEVICE)",
        "    model     = build_model(\n"
        "        use_cbam=not getattr(args, 'no_cbam', False)).to(DEVICE)",
    ),
    (
        "    p.add_argument('--resume', action='store_true',",
        "    p.add_argument('--no_cbam', action='store_true',\n"
        "                   help='Ablation: replace all seven CBAM modules '\n"
        "                        'with identity (21,769 -> 19,291 params).')\n"
        "    p.add_argument('--resume', action='store_true',",
    ),
])

# ------------------------------------------------- evaluate / sharpness
LOADER_OLD = (
    "    model = ZeroDCECBAM(num_iters=8, channels=32).to(device)\n"
    "    ckpt = torch.load(weights, map_location=device, weights_only=False)\n"
    "    state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt\n"
)
LOADER_NEW = (
    "    ckpt = torch.load(weights, map_location=device, weights_only=False)\n"
    "    state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt\n"
    "    # Older checkpoints named the output layer 'out'.\n"
    "    if any(k.startswith('out.') for k in state):\n"
    "        state = {('curve_out.' + k[4:] if k.startswith('out.') else k): v\n"
    "                 for k, v in state.items()}\n"
    "    # A no-CBAM ablation checkpoint has no cb*.* keys; match it.\n"
    "    use_cbam = any(k.startswith('cb1.') for k in state)\n"
    "    if not use_cbam:\n"
    "        print('  [ablation checkpoint: CBAM disabled]')\n"
    "    model = ZeroDCECBAM(num_iters=8, channels=32,\n"
    "                        use_cbam=use_cbam).to(device)\n"
)
for f in ('src/evaluate.py', 'src/measure_sharpness.py'):
    patch(f, [(LOADER_OLD, LOADER_NEW)])

# ---------------------------------------------------------------- verify
sys.path.insert(0, ROOT)
try:
    from src.model import build_model
    a = sum(p.numel() for p in build_model().parameters())
    b = sum(p.numel() for p in build_model(use_cbam=False).parameters())
    print(f'\nwith CBAM   : {a:,}\nwithout CBAM: {b:,}\ndifference  : {a-b:,}')
    assert a == 21769 and b == 19291, 'unexpected parameter counts'
    print('\nPatch verified.')
except Exception as exc:                                  # pragma: no cover
    print(f'\nVerification skipped/failed: {exc}')

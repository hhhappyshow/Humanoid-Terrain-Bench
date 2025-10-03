# A small utility to inspect a PyTorch checkpoint and compare shapes with N1 config
import sys
import torch
from legged_gym.envs.N1.n1_fix import N1FixCfg

if len(sys.argv) < 2:
    print("Usage: python check_ckpt_compat.py /path/to/checkpoint.pth")
    sys.exit(1)

ckpt_path = sys.argv[1]
print("Loading checkpoint:", ckpt_path)
ckpt = torch.load(ckpt_path, map_location='cpu')
# support both full dict and state_dict
sd = None
if isinstance(ckpt, dict):
    if 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    elif 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    else:
        # try to find a nested dict of tensors
        candidates = {k:v for k,v in ckpt.items() if isinstance(v, dict)}
        if len(candidates) == 1:
            sd = list(candidates.values())[0]
        else:
            # fallback: use ckpt itself
            sd = ckpt
else:
    sd = ckpt

print('\n==== N1 CONFIG ====')
print('num_observations (cfg) =', N1FixCfg.env.num_observations)
print('num_actions (cfg)      =', N1FixCfg.env.num_actions)

print('\n==== CHECKPOINT TENSOR SHAPES (showing up to 200 keys) ====')
count = 0
for k, v in sd.items():
    if isinstance(v, torch.Tensor):
        print(f"{k}: {tuple(v.shape)}")
        count += 1
        if count >= 200:
            print('... (truncated)')
            break

# try to heuristically find actor input/output layers
print('\n==== HEURISTIC ACTOR/OUTPUT LAYER SEARCH ====')
potential_in = []
potential_out = []
for k, v in sd.items():
    if isinstance(v, torch.Tensor):
        # weights often have 2 dims
        if v.dim() == 2:
            name = k.lower()
            if 'actor' in name or 'policy' in name:
                # weights shaped (out, in) or (in, out)
                potential_in.append((k, tuple(v.shape)))
            if 'action' in name or 'action_head' in name or 'actor_head' in name:
                potential_out.append((k, tuple(v.shape)))

print('Found candidate actor-related tensors (first 20):')
for item in potential_in[:20]:
    print('  in-cand:', item)
for item in potential_out[:20]:
    print('  out-cand:', item)

print('\nIf dimensions mismatch, you can either adapt the checkpoint (pad/trim) or pad observations at env side.')
print('Exit.')

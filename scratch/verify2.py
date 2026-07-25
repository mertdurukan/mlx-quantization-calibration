import mlx.nn as nn, mlx.core as mx
from mlx.utils import tree_flatten

def probe(mode, bits=4, g=64):
    mx.random.seed(0)
    lin = nn.Linear(256, 256)
    nn.quantize(lin, group_size=g, bits=bits, mode=mode)
    mx.eval(lin.parameters())
    attrs = {k: getattr(lin, k, None) for k in ["bits", "group_size", "mode"]}
    w = dict(tree_flatten(lin.parameters()))
    s = round(sum(float(mx.sum(v.astype(mx.float32))) for v in w.values()), 2)
    return attrs, sorted(w), s

print("=== MOD PARMAK IZI ===")
for m in ["affine", "mxfp4", "nf4", "mxfp8", "banana"]:
    try:
        a, ks, s = probe(m)
        print(f"  {m:8} attrs={a} sum={s}")
    except Exception as e:
        print(f"  {m:8} HATA {type(e).__name__}: {str(e)[:60]}")

print("\n=== BIT PARMAK IZI (affine) ===")
for b in [2,3,4,5,6,8]:
    try:
        a, ks, s = probe("affine", bits=b)
        print(f"  {b}-bit attrs={a} sum={s}")
    except Exception as e:
        print(f"  {b}-bit HATA {type(e).__name__}: {str(e)[:60]}")

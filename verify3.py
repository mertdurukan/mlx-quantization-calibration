import mlx.nn as nn, mlx.core as mx
from mlx.utils import tree_flatten

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256, 256)

def probe(mode, bits=4, g=64):
    mx.random.seed(0)
    net = Net()
    before = type(net.fc).__name__
    nn.quantize(net, group_size=g, bits=bits, mode=mode)
    mx.eval(net.parameters())
    after = type(net.fc).__name__
    keys = sorted(k for k, _ in tree_flatten(net.parameters()))
    w = dict(tree_flatten(net.parameters()))
    s = round(sum(float(mx.sum(v.astype(mx.float32))) for v in w.values()), 3)
    return before, after, keys, s

print("=== SANITY: kuantizasyon GERCEKTEN oluyor mu? ===")
b, a, ks, s = probe("affine")
print(f"  {b} -> {a}")
print(f"  keys: {ks}")
print(f"  {'GERCEK' if a != b else 'HALA NO-OP'}")

print("\n=== MOD ===")
for m in ["affine", "mxfp4", "nf4", "mxfp8", "banana"]:
    try:
        _, a, ks, s = probe(m)
        print(f"  {m:8} -> {a:16} sum={s:10} keys={len(ks)}")
    except Exception as e:
        print(f"  {m:8} REDDEDILDI: {type(e).__name__}: {str(e)[:50]}")

print("\n=== BITS (affine) ===")
for bt in [2,3,4,5,6,8]:
    try:
        _, a, ks, s = probe("affine", bits=bt)
        print(f"  {bt}-bit -> {a:16} sum={s}")
    except Exception as e:
        print(f"  {bt}-bit REDDEDILDI: {str(e)[:50]}")

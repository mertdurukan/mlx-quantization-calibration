"""Mutasyon: sacma bir mode da geciyorsa, test bos demektir."""
import mlx.nn as nn, mlx.core as mx

print("=== MUTASYON: sacma modlar ===")
for m in ["banana", "xyz123", ""]:
    try:
        nn.quantize(nn.Linear(256,256), group_size=64, bits=4, mode=m)
        print(f"  {m!r:10} OK  <-- TEST BOS, mode dogrulanmiyor")
    except Exception as e:
        print(f"  {m!r:10} reddedildi: {type(e).__name__}")

print("\n=== GERCEK FARK: mod agirliklari degistiriyor mu? ===")
def fingerprint(mode, bits=4):
    mx.random.seed(0)
    lin = nn.Linear(256, 256)
    nn.quantize(lin, group_size=64, bits=bits, mode=mode)
    ks = sorted(k for k, _ in mx.utils.tree_flatten(lin.parameters()))
    w = dict(mx.utils.tree_flatten(lin.parameters()))
    total = sum(float(mx.sum(v.astype(mx.float32))) for v in w.values())
    return ks, round(total, 3)

for m in ["affine", "mxfp4", "nf4", "banana"]:
    try:
        ks, s = fingerprint(m)
        print(f"  {m:8} keys={ks} sum={s}")
    except Exception as e:
        print(f"  {m:8} HATA: {type(e).__name__}")

print("\n=== QUANT_RECIPES ===")
from mlx_lm.convert import QUANT_RECIPES
print(" ", QUANT_RECIPES if not isinstance(QUANT_RECIPES, dict) else list(QUANT_RECIPES))

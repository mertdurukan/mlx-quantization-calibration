"""Kesif: hangi mode/bits gercekten calisiyor? Denenerek kanitlanir."""
import mlx.nn as nn
import mlx.core as mx

def tiny():
    return nn.Linear(256, 256)

print("=== MODE ===")
for m in ["affine", "mxfp4", "nf4", "mxfp8", "dynamic", "int", "symmetric"]:
    try:
        nn.quantize(tiny(), group_size=64, bits=4, mode=m)
        print(f"  {m:12} OK")
    except Exception as e:
        msg = str(e).split(chr(10))[0][:70]
        print(f"  {m:12} -- {type(e).__name__}: {msg}")

print("\n=== BITS (mode=affine, group=64) ===")
for b in [2, 3, 4, 5, 6, 8]:
    try:
        nn.quantize(tiny(), group_size=64, bits=b, mode="affine")
        print(f"  {b}-bit  OK")
    except Exception as e:
        print(f"  {b}-bit  -- {str(e).split(chr(10))[0][:60]}")

print("\n=== GROUP SIZE (4-bit) ===")
for g in [32, 64, 128]:
    try:
        nn.quantize(tiny(), group_size=g, bits=4, mode="affine")
        print(f"  group={g:4} OK")
    except Exception as e:
        print(f"  group={g:4} -- {str(e).split(chr(10))[0][:60]}")

print("\n=== mlx_lm icinde isimli quant semalari ===")
import mlx_lm, pkgutil, importlib
found = []
for mod in ["mlx_lm.convert", "mlx_lm.utils", "mlx_lm.quant"]:
    try:
        m = importlib.import_module(mod)
        for n in dir(m):
            if "predicate" in n.lower() or "recipe" in n.lower() or "mixed" in n.lower():
                found.append(f"{mod}.{n}")
    except Exception:
        pass
print("  " + (", ".join(found) if found else "bulunamadi"))

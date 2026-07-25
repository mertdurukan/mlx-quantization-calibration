"""Kesif 2: hangi base modeller var, kendi donusturmemiz calisiyor mu?"""
import time, os, shutil
from mlx_lm import load, convert

# Aday base modeller (bf16 / fp16 kaynak, kendi kuantizasyonumuzu yapacagiz)
CANDIDATES = [
    "mlx-community/Qwen2.5-0.5B-Instruct-bf16",
    "mlx-community/Qwen2.5-1.5B-Instruct-bf16",
    "mlx-community/Llama-3.2-1B-Instruct-bf16",
    "mlx-community/Llama-3.2-3B-Instruct-bf16",
    "mlx-community/Qwen2.5-3B-Instruct-bf16",
]

print("=== BASE MODEL ERISIM TESTI (yalnizca metadata) ===")
from huggingface_hub import model_info
for m in CANDIDATES:
    try:
        info = model_info(m)
        size = sum(s.size or 0 for s in (info.siblings or [])) / 1e9
        print(f"  OK   {m:52} ~{size:.2f} GB")
    except Exception as e:
        print(f"  YOK  {m:52} {type(e).__name__}")

print("\n=== KENDI DONUSTURMEMIZ CALISIYOR MU? (0.5B, 4-bit affine) ===")
OUT = "/tmp/mlxq_test_4bit"
shutil.rmtree(OUT, ignore_errors=True)
t = time.time()
try:
    convert(
        hf_path="mlx-community/Qwen2.5-0.5B-Instruct-bf16",
        mlx_path=OUT,
        quantize=True,
        q_bits=4,
        q_group_size=64,
        q_mode="affine",
    )
    dt = time.time() - t
    sz = sum(os.path.getsize(os.path.join(OUT,f)) for f in os.listdir(OUT)) / 1e6
    print(f"  DONUSTURME OK  {dt:.1f} sn, cikti {sz:.0f} MB")
    model, tok = load(OUT)
    print(f"  YUKLEME OK     {type(model).__name__}")
    print("\n  KAPI: kendi kuantizasyonumuzu uretip yukleyebiliyoruz.")
except Exception as e:
    print(f"  HATA: {type(e).__name__}: {str(e)[:200]}")

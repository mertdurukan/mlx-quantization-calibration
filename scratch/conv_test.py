import os, time, shutil
from huggingface_hub import snapshot_download
from mlx_lm import load, convert

SRC = "mlx-community/Qwen2.5-0.5B-Instruct-bf16"
print("1. Tam snapshot indiriliyor...")
p = snapshot_download(SRC)
print("   OK:", p)
print("   dosyalar:", sorted(os.listdir(p)))

OUT = "/tmp/mlxq_test_4bit"
shutil.rmtree(OUT, ignore_errors=True)
print("\n2. Donusturuluyor (4-bit affine g=64)...")
t = time.time()
convert(hf_path=SRC, mlx_path=OUT, quantize=True, q_bits=4, q_group_size=64, q_mode="affine")
dt = time.time() - t
sz = sum(os.path.getsize(os.path.join(OUT,f)) for f in os.listdir(OUT)) / 1e6
print(f"   OK {dt:.1f} sn, {sz:.0f} MB")

print("\n3. Yukleniyor...")
m, tok = load(OUT)
print("   OK", type(m).__name__)
print("\nKAPI: kendi kuantizasyonumuzu uretip yukleyebiliyoruz.")

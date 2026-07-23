"""PREREG 4.6.6 -- determinizm sozlesmesi + mutasyon kaniti."""
import shutil
import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm import load, convert
from datasets import load_dataset

NL = chr(10)
SRC = "mlx-community/Qwen2.5-1.5B-Instruct-bf16"
N = 20

d = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

def logprobs(path, n=N, noise=0.0):
    m, tok = load(path)
    out = []
    for i in range(n):
        ex = d[i]
        labs = ex["choices"]["label"]
        opts = NL.join(l + ") " + t for l, t in zip(labs, ex["choices"]["text"]))
        ids = tok.encode("Question: " + ex["question"] + NL + opts + NL + "Answer:")
        lg = m(mx.array([ids]))[0, -1, :]
        if noise:
            lg = lg + mx.random.normal(lg.shape) * noise
        lp = lg - mx.logsumexp(lg)
        out.append(tuple(float(lp[tok.encode(" " + L)[0]]) for L in labs))
    return out

print("1. Model hazirlaniyor (1.5B, 4-bit affine g=64)...")
snapshot_download(SRC)
OUT = "/tmp/mlxq_det"
shutil.rmtree(OUT, ignore_errors=True)
convert(hf_path=SRC, mlx_path=OUT, quantize=True, q_bits=4, q_group_size=64, q_mode="affine")

print("2. Ayni hucre iki kez kosuluyor...")
a = logprobs(OUT)
b = logprobs(OUT)
same = (a == b)
print(f"   bit-bit ayni: {same}")
if not same:
    diffs = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    print(f"   {len(diffs)}/{N} item farkli. ilk fark: {diffs[0]}")

print("3. MUTASYON: kasten gurultu eklenirse test yakaliyor mu?")
c = logprobs(OUT, noise=1e-6)
caught = (a != c)
print(f"   gurultulu kosu farkli algilandi: {caught}")

print()
if same and caught:
    print("SOZLESME GECTI: inference deterministik VE test bunu dogrulayabiliyor.")
elif same and not caught:
    print("TEST BOS: gurultu bile yakalanmadi, karsilastirma anlamsiz.")
else:
    print("DETERMINIZM YOK: PREREG 4.6.6 uyarinca tasarim revize edilmeli.")

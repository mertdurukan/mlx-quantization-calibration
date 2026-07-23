"""Kapinin ikinci yarisi: ayni modeli farkli kuantizasyonda yukleyip karsilastirabiliyor muyuz?"""
import math
import mlx.core as mx
from mlx_lm import load

PROMPT = "Question: What is 2+2?\nA) 3\nB) 4\nC) 5\nD) 6\nAnswer:"
MODELS = {
    "bf16":  "mlx-community/Qwen2.5-0.5B-Instruct-bf16",
    "4bit":  "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
}

def probs(model_id):
    model, tok = load(model_id)
    ids = tok.encode(PROMPT)
    lg = model(mx.array([ids]))[0, -1, :]
    lp = lg - mx.logsumexp(lg)
    out = {}
    for L in "ABCD":
        t = tok.encode(f" {L}")
        out[L] = math.exp(float(lp[t[0]]))
    return out

rows = {}
for name, mid in MODELS.items():
    print(f"yukleniyor: {name}")
    try:
        rows[name] = probs(mid)
        print(f"   OK")
    except Exception as e:
        print(f"   HATA: {type(e).__name__}: {e}")

print("\n%-6s %8s %8s %8s %8s" % ("", "A", "B", "C", "D"))
for name, r in rows.items():
    print("%-6s %8.4f %8.4f %8.4f %8.4f" % (name, r["A"], r["B"], r["C"], r["D"]))

if len(rows) == 2:
    print("\nKAPI TAM GECTI: iki kuantizasyon seviyesi karsilastirilabiliyor.")
else:
    print("\nEksik: bf16 yuklenemedi. mlx_lm.convert ile kendimiz uretmeliyiz.")

import time, math
import mlx.core as mx
from mlx_lm import load
from datasets import load_dataset

m, tok = load("/tmp/mlxq_test_4bit")
d = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

def ask(ex):
    labs = ex["choices"]["label"]; txts = ex["choices"]["text"]
    opts = chr(10).join(f"{l}) {t}" for l, t in zip(labs, txts))
    p = f"Question: {ex[chr(113)+chr(117)+chr(101)+chr(115)+chr(116)+chr(105)+chr(111)+chr(110)]}{chr(10)}{opts}{chr(10)}Answer:"
    ids = tok.encode(p)
    lg = m(mx.array([ids]))[0, -1, :]
    lp = lg - mx.logsumexp(lg)
    out = {}
    for L in labs:
        t = tok.encode(f" {L}")
        out[L] = float(lp[t[0]])
    return out

N = 50
t0 = time.time()
correct = 0
for i in range(N):
    ex = d[i]
    sc = ask(ex)
    pred = max(sc, key=sc.get)
    correct += (pred == ex["answerKey"])
dt = time.time() - t0
print(f"{N} soru: {dt:.1f} sn -> {dt/N*1000:.0f} ms/soru")
print(f"dogruluk: {correct}/{N}")
print(f"1000 soru tahmini: {dt/N*1000/60:.1f} dk")

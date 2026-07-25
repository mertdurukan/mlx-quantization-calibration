import time
import mlx.core as mx
from mlx_lm import load
from datasets import load_dataset
NL = chr(10)
d = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
def bench(path, n=30):
    m, tok = load(path)
    t0 = time.time()
    for i in range(n):
        ex = d[i]
        labs = ex["choices"]["label"]
        opts = NL.join(l + ") " + t for l, t in zip(labs, ex["choices"]["text"]))
        ids = tok.encode("Question: " + ex["question"] + NL + opts + NL + "Answer:")
        lg = m(mx.array([ids]))[0, -1, :]
        lp = lg - mx.logsumexp(lg)
        _ = {L: float(lp[tok.encode(" " + L)[0]]) for L in labs}
    return (time.time() - t0) / n * 1000
print(f"0.5B tekrar : {bench(chr(47)+chr(116)+chr(109)+chr(112)+chr(47)+chr(109)+chr(108)+chr(120)+chr(113)+chr(95)+chr(116)+chr(101)+chr(115)+chr(116)+chr(95)+chr(52)+chr(98)+chr(105)+chr(116)):5.0f} ms/soru  (ilk olcum 52)")
print(f"3B tekrar   : {bench(chr(47)+chr(116)+chr(109)+chr(112)+chr(47)+chr(109)+chr(108)+chr(120)+chr(113)+chr(95)+chr(51)+chr(66)+chr(95)+chr(52)+chr(98)+chr(105)+chr(116)):5.0f} ms/soru  (ilk olcum 693)")

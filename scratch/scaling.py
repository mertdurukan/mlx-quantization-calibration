import time, shutil
from huggingface_hub import snapshot_download
import mlx.core as mx
from mlx_lm import load, convert
from datasets import load_dataset

NL = chr(10)
d = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

def bench(path, n=30):
    m, tok = load(path)
    t0 = time.time()
    hit = 0
    for i in range(n):
        ex = d[i]
        labs = ex["choices"]["label"]
        txts = ex["choices"]["text"]
        opts = NL.join(l + ") " + t for l, t in zip(labs, txts))
        p = "Question: " + ex["question"] + NL + opts + NL + "Answer:"
        ids = tok.encode(p)
        lg = m(mx.array([ids]))[0, -1, :]
        lp = lg - mx.logsumexp(lg)
        sc = {L: float(lp[tok.encode(" " + L)[0]]) for L in labs}
        hit += (max(sc, key=sc.get) == ex["answerKey"])
    return (time.time() - t0) / n * 1000, hit, n

ms, h, n = bench("/tmp/mlxq_test_4bit")
print(f"0.5B 4bit: {ms:5.0f} ms/soru  dogruluk {h}/{n}")

for tag, src in [("1.5B", "mlx-community/Qwen2.5-1.5B-Instruct-bf16"), ("3B", "mlx-community/Qwen2.5-3B-Instruct-bf16")]:
    print(f"{tag}: indiriliyor...")
    snapshot_download(src)
    OUT = "/tmp/mlxq_" + tag + "_4bit"
    shutil.rmtree(OUT, ignore_errors=True)
    convert(hf_path=src, mlx_path=OUT, quantize=True, q_bits=4, q_group_size=64, q_mode="affine")
    ms, h, n = bench(OUT)
    print(f"{tag} 4bit: {ms:5.0f} ms/soru  dogruluk {h}/{n}")

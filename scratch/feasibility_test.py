"""Fizibilite (Kural 2): MLX ile MC soruda log-prob okunabiliyor mu?"""
import math
import mlx.core as mx
from mlx_lm import load

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

print(f"1. Model yukleniyor: {MODEL}")
model, tokenizer = load(MODEL)
print("   OK yuklendi")

prompt = "Question: What is 2+2?\nA) 3\nB) 4\nC) 5\nD) 6\nAnswer:"
tokens = tokenizer.encode(prompt)
print(f"2. Prompt tokenize edildi: {len(tokens)} token")

out = model(mx.array([tokens]))
print(f"   logits shape: {out.shape}")
logits = out[0, -1, :]
logprobs = logits - mx.logsumexp(logits)

print("3. Secenek log-prob'lari:")
results = {}
for letter in ["A", "B", "C", "D"]:
    got = False
    for variant in [f" {letter}", letter]:
        try:
            ids = tokenizer.encode(variant, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(variant)
        if ids:
            lp = float(logprobs[ids[0]])
            results[letter] = lp
            print(f"   {variant!r:5} tok={ids[0]:6d}  logprob={lp:8.3f}  prob={math.exp(lp):.5f}")
            got = True
            break
    if not got:
        print(f"   {letter}: TOKENIZE EDILEMEDI")

if results:
    best = max(results, key=results.get)
    print(f"\n4. En yuksek: {best}  (dogru cevap: B)")
    print("KAPI GECTI" if best == "B" else "log-prob okundu ama model B demedi")
else:
    print("\nKAPI GECMEDI: log-prob okunamadi")

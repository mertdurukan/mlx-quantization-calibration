# Fizibilite Testi — Kural 2 Kapısı

> **Bu test geçmeden ön-kayıt yazılmaz.** ML çalışmasında bu kapı atlandı ve iki sapma doğdu.

## Amaç

Tek bir soruyu yanıtla: **MLX ile bir modeli yükleyip, çoktan-seçmeli bir soruda cevap
seçeneklerinin log-probability'sini okuyabiliyor muyuz?**

Çalışma tamamen bunun üzerine kurulu. Okuyamıyorsak, konu Mac'te yapılamaz ve açı değişir.

## Ortam (hazır)

```bash
cd ~/github-projects/mlx-quantization-calibration
source .venv/bin/activate
```

Prompt'ta `(.venv)` ve dizin adı görünmeli. Kurulu: `mlx-lm 0.31.3`, Python 3.14.

⚠️ `feasibility_test.py` yanlışlıkla home dizinine düştü. Taşı:
```bash
mv ~/feasibility_test.py . 2>/dev/null; ls -la feasibility_test.py
```

## Test taslağı

```python
"""Fizibilite: MLX ile MC soruda log-prob okunabiliyor mu?"""
import math
import mlx.core as mx
from mlx_lm import load

print("1. Model yükleniyor (ilk sefer indirir)...")
model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
print("   ✓ yüklendi")

prompt = "Question: What is 2+2?\nA) 3\nB) 4\nC) 5\nD) 6\nAnswer:"
tokens = tokenizer.encode(prompt)
print(f"2. {len(tokens)} token")

logits = model(mx.array([tokens]))[0, -1, :]
logprobs = logits - mx.logsumexp(logits)   # log-softmax

print("3. Seçenek log-prob'ları:")
for letter in ["A", "B", "C", "D"]:
    for variant in [f" {letter}", letter]:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if ids:
            lp = float(logprobs[ids[0]])
            print(f"   {variant!r:6} → logprob={lp:.3f}  prob={math.exp(lp):.4f}")
            break

print("\n✓ FIZIBILITE GECTI")
```

## Başarı kriteri

- Dört seçeneğin log-prob'u okunuyor
- **"B" (doğru cevap) en yüksek olmalı** — model 2+2=4 biliyor
- Hata yok

## Not — MLX API hızlı değişiyor

`mlx-lm 0.31.3`'te fonksiyon adları/imzaları farklı olabilir. İlk denemede tutmayabilir;
normal. Hata mesajını oku, API'ye göre düzelt. **Testin amacı değişmez:** log-prob
okunabiliyor mu?

## Test geçerse — sonraki keşif adımı (yine Kural 2, hâlâ ön-kayıt öncesi)

Ön-kaydı yazmadan önce şunlar da doğrulanmalı:

1. **Hangi kuantizasyon seçenekleri gerçekten mevcut?**
   `mlx_lm.convert` hangi bit genişliklerini ve modları destekliyor? (fp16, 8-bit,
   uniform 4-bit, mixed-bit, `dynamic_quant` — hepsi var mı, adları ne?)
2. **Hangi modeller mlx-community'de hem fp16 hem kuantize mevcut?** Yoksa kendimiz mi
   dönüştüreceğiz? (Kendi dönüştürmemiz daha kontrollü olur — aynı kaynak, tek değişken.)
3. **Hangi MC benchmark?** (BoolQ / PIQA / ARC-Challenge / HellaSwag gibi, log-prob'dan
   güven okunabilen türden.) İndirilebiliyor mu, boyutu ne?
4. **Bellek:** 24GB unified'da hangi model boyutuna kadar fp16 sığıyor?

Bu dördü yanıtlanmadan ön-kayıt yazılırsa, ML'deki "havuz 8 dataset verdi" sürprizinin
aynısı yaşanır.

## Test geçmezse

Açıyı değiştir. Seçenekler:
- Farklı bir MLX ölçümü (log-prob yerine başka bir güven sinyali)
- Konuyu tamamen bırak, ML çalışmasının dağıtımına odaklan

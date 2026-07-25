# DURUM — Canlı Durum Dosyası

> **Her oturum sonunda güncellenir.** Bu dosya, "neredeyiz" sorusunun tek cevabıdır.
> Son güncelleme: 2026-07-25

---

## Faz

```
[x] FAZ 0 — Fizibilite kapısı           TAMAMLANDI
[x] FAZ 1 — Keşif (eksenler, veri, hız) TAMAMLANDI
[x] FAZ 2 — Ön-kayıt donduruldu         TAMAMLANDI  (commit c5ea71c)
[ ] FAZ 3 — Implementasyon              ← ŞU AN BURADAYIZ
[ ] FAZ 4 — Pilot + tam koşu
[ ] FAZ 5 — Analiz (ön-kayıtlı tablolar)
[ ] FAZ 6 — Makale + dağıtım
```

## Şu an

**FAZ 3, Görev 1 tamamlandı.** İskelet ayakta, prompt şablonu donduruldu, `src/config.py` tek
gerçek kaynak olarak yazıldı. Henüz gerçek metrik/benchmark/quantize kodu yok.

Repoda şu an olanlar: `src/config.py` (+ `__init__.py`), `prompts/mc_letter.txt` (donmuş,
SHA-256 `config.PROMPT_SHA256` ile korunuyor), `tests/test_prompt_frozen.py`,
`tests/test_determinism.py` (taşındı, PREREG §4.6.6), `requirements.txt` +
`requirements.lock.txt`, `Makefile`, `.gitignore`, `results/{cells,meta,tables,figures}`
iskeleti, `scratch/` (14 keşif script'i taşındı, hâlâ commit'te — bkz. GECMIS.md).

Sıradaki görev: `YAPILACAKLAR.md` → **Görev 2** (metrik testleri — önce test, implementasyon YOK,
kullanıcı onayından sonra Görev 3).

## Son oturumda ne oldu (2026-07-25)

1. Dizin iskeleti oluşturuldu: `src/`, `tests/`, `prompts/`, `results/{cells,meta,tables,figures}`,
   `scratch/`
2. `src/config.py` SPEC §2'den yazıldı; ek olarak `PROMPT_SHA256` eklendi (SPEC §9 Changelog'a
   ve GECMIS.md'ye işlendi — SPEC §2'nin orijinal listesinde yoktu, görev tanımı gerektirdi)
3. `prompts/mc_letter.txt` donduruldu (PREREG §4.4 format), SHA-256 kaydedildi
4. `tests/test_prompt_frozen.py` yazıldı ve geçiyor
5. `requirements.txt` (5 doğrudan bağımlılık, `pytest` dahil) + `requirements.lock.txt`
   (`pip freeze`, 56 paket) yazıldı
6. `Makefile` (SPEC §6 hedefleri), `.gitignore` yazıldı
7. 14 keşif script'i `scratch/`'e taşındı (silinmedi, hâlâ commit'te — gitignore yalnızca yeni
   dosyalara uygulanıyor); `test_determinism.py` → `tests/test_determinism.py`
8. Kabul kriteri çalıştırıldı ve doğrulandı: `make test` → 1 passed, `1000 14` yazdı
9. **Açık bulgu kaydedildi:** `tests/test_determinism.py` pytest formatında değil (assert yok,
   her `make test`'te gerçek model conversion'ı tetikliyor) — YAPILACAKLAR § AÇIK BULGULAR

## Önceki oturum (2026-07-24)

Fizibiliteden ön-kayda kadar tüm zincir tamamlandı:

1. MLX'in log-prob okuyabildiği kanıtlandı (4 seçenek, kütlenin %99'u seçeneklerde)
2. İki kuantizasyon seviyesinin karşılaştırılabildiği kanıtlandı
3. **Üç boş test yakalandı ve düzeltildi** (aşağıda, GECMIS.md'de detay)
4. Kuantizasyon eksenleri **çalıştırılarak** doğrulandı: bits 2-8 affine, mxfp4/mxfp8 → g=32
   zorunlu, `nf4` yok, `dynamic_quant` yok, 4 mixed recipe var
5. Kendi dönüştürme pipeline'ımızın çalıştığı kanıtlandı (2 sn, tek bf16 kaynaktan)
6. Model havuzu ve hız ölçüldü: 0.5B %33 (şans seviyesi → ana ızgaradan çıkarıldı),
   1.5B %73 @127ms, 3B %73 @193ms
7. Termal throttling **yok** (0.5B tekrar ölçümde 50ms, stabil)
8. Benchmark'lar doğrulandı: ARC-Challenge + MMLU (aynı protokol), HellaSwag/PIQA elendi
9. **Ön-kayıt donduruldu ve push'landı** — `c5ea71c`, 2026-07-24
10. **Determinizm sözleşmesi geçti** ve mutasyonla kanıtlandı (1e-6 gürültü yakalandı)

## Kritik açık noktalar

- `src/` hiç yok — implementasyon sıfırdan başlayacak
- `prompts/mc_letter.txt` **henüz yazılmadı** ve donmadı. İlk koşudan önce donmalı.
- MMLU stratified örnekleme mantığı henüz yazılmadı (57 konu, deterministik olmalı)
- Uygunluk kapısı (bf16 ≥ %50) sırası `runner.py`'de mekanik olarak zorlanmalı

## Bütçe hatırlatması

Tam ızgara ≈ 140 hücre (5 model × 14 koşul × 2 benchmark). Tahmini **4–6 saat**, resumable.
Disk: `convert → evaluate → delete`, HF cache ≈ 20 GB.

**Pilotta tek hücre süresini ölç, 140 ile çarp. 8 saati aşarsa DUR ve bildir.**

---

## Oturum günlüğü

| Tarih | Ne yapıldı | Commit |
|---|---|---|
| 2026-07-24 | Fizibilite kapısı, eksen keşfi, ön-kayıt, determinizm | `ffa07c7` … `c5ea71c` |
| 2026-07-25 | Görev 1 — ortam iskeleti, config, prompt dondurma | (bu oturum) |
| | | |

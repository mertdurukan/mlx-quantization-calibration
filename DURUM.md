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

**FAZ 3, Görev 1-2-3-4-5 tamamlandı. Sırada Görev 6 (`tests/test_no_leakage.py` + MUTASYON KANITI).**
`src/quantize.py` (`build`, `teardown`) ve `src/measure.py` (`run_cell`) yazıldı. `pytest tests/ -v`
→ **25 passed** (yeni test dosyası yok — Görev 5'in SPEC talimatı test-önce gerektirmiyordu).

Repoda şu an olanlar: `src/config.py`, `src/metrics.py` (Görev 3), `src/benchmarks.py` (Görev 4),
`src/quantize.py` + `src/measure.py` (Görev 5, tam implementasyon, dört mod da — bf16, affine,
mxfp4, recipe — gerçek dönüşümle fonksiyonel doğrulandı), `prompts/mc_letter.txt` (donmuş,
SHA-256 `config.PROMPT_SHA256` ile korunuyor), `tests/test_prompt_frozen.py`,
`tests/test_determinism.py` (PREREG §4.6.6, gerçek `assert`'li 2 test), `tests/test_metrics.py`
(13 test), `tests/test_benchmarks.py` (9 test), `requirements.txt` + `requirements.lock.txt`
(`statsmodels`, `scipy` dahil), `Makefile`, `.gitignore`, `results/{cells,meta,tables,figures}`
iskeleti, `scratch/` (14 keşif script'i, hâlâ commit'te). `CLAUDE.md` Adım 2'deki görev-numarası
taraması bu oturumda **ikinci kez** işe yaradı: Görev 5'e etiketli açık bulgu (`mixed_*` recipe
çağrı biçimi) tarama sırasında yakalandı ve implementasyona başlamadan **çalıştırılarak**
çözüldü — bkz. aşağı.

**DUR noktası yok şu an** — Görev 5 kullanıcı onayı gerektirmiyordu. Sıradaki iş Görev 6, ve
Görev 6 PROTOKOL Kural 4 gereği **mutasyon kanıtı** içeriyor — atlanamaz.

## Son oturumda ne oldu (2026-07-25, Görev 5 — quantize.py + measure.py)

1. **Adım 2 taraması Görev 5'i engelleyen bir açık bulgu buldu:** "`mixed_*` recipe'lerin nasıl
   çağrıldığı doğrulanmadı · Engellediği görev: 5". `mlx_lm/convert.py` kaynağı okunarak ve
   cache'teki `Qwen2.5-0.5B-Instruct-bf16` ile **gerçek bir `convert()` çağrısı** yapılarak
   çözüldü: `quant_predicate` doğrudan `condition_tag` string'i (`mlx_lm.QUANT_RECIPES` ile
   birebir eşleşiyor), `q_mode="affine"` zorunlu, `q_group_size=None`/`q_bits=None`. `mixed_2_6`
   ile dönüştürülen modelin `config.json`'ında 169 katmanda yalnızca `{2, 6}` bit görüldü, model
   `load()`+`generate()` ile gerçekten çalıştırıldı. Detay: GECMIS.md.
2. `src/quantize.py` yazıldı: `build(model_key, condition_tag, out_dir)` dört dalı yönetiyor
   (bf16 → cache yolu, `effective_bits=16.0`; affine/mxfp4 → doğrudan bits/group_size; recipe →
   `quant_predicate=`). `effective_bits`, `convert()`'in stdout'undan (`"Quantized model with
   X bits per weight."`) regex ile parse ediliyor — SPEC'in "varsayılmaz" kuralı. `teardown`
   dizini siler.
3. `src/measure.py` yazıldı: `run_cell(model_path, items)`. Warmup semantiği SPEC §4'teki meta
   şemasından (`n_items_scored`/`n_warmup_discarded` ayrı alanlar) çıkarıldı: ilk `N_WARMUP`
   item önce çalıştırılıp atılıyor, **sonra tüm** `items` (1000'in tamamı, ilk 20 dahil)
   skorlanıyor — yani skorlanan sayı her zaman `N_ITEMS`, warmup ek maliyet. Güven, seçenek
   harfi token'larının (` A`, ` B`, ...) log-prob'ları üzerinden softmax ile hesaplanıyor; tek
   item hatası `status="failed"` ile hücreyi düşürmüyor.
4. Kabul kriteri çalıştırıldı: `build('qwen2.5-1.5b','affine_b4_g64',...)` →
   `effective_bits: 4.501 | nominal 4 ile ayni mi: False`. Ek fonksiyonel doğrulama: bf16
   (cache yolu, 16.0), mxfp4_b4_g32 (4.252), mixed_2_6 (2.937, `build` üzerinden de aynı sonuç);
   `measure.run_cell` cache'teki 0.5B + 30 ARC item ile çalıştırıldı → 30 satır, 0 failed,
   doğruluk ~%37. `pytest tests/ -q` → 25 passed (regresyon yok).

## Önceki oturum (2026-07-25, Görev 4 — benchmarks.py)

1. **Adım 2 taraması Görev 4'ü engelleyen bir açık bulgu buldu:** "MMLU'da seçenek sayısı her
   zaman 4 mü, doğrulanmadı · Engellediği görev: 4". İmplementasyona başlamadan önce
   **çalıştırılarak** çözüldü: `cais/mmlu` `all`/test (14042 satır) tarandı → hepsi tam 4
   seçenekli. `allenai/ai2_arc` ARC-Challenge test (1172 satır) de tarandı → seçenek sayısı
   **{3, 4, 5} arasında değişiyor**. Bu, SPEC'teki `Item.options: list[str]` tasarımının
   (sabit 4 değil) doğru olduğunu kanıtladı. Detay: GECMIS.md.
2. `src/benchmarks.py` yazıldı: `Item` (frozen dataclass), `load_items(benchmark)`.
   - ARC: `choices.label` içinde `answerKey`'in indeksini bularak `answer_idx` üretiliyor
     (bazı ARC satırlarında etiketler `"1","2","3","4"` — harf değil, doğrulanarak görüldü).
   - MMLU: konu başına orantılı kota (`N_ITEMS * count_s // total`), kalan `numpy` ile
     `seed=config.SEED` kullanılarak konu adı sırasına göre dağıtılıyor; her konu içinde
     `rng.choice(..., replace=False)` ile örnekleniyor. `item_id` doğal id olmadığı için
     `mmlu_{subject}_{orijinal_indeks}` olarak üretiliyor (deterministik, tekil).
   - Harfler her iki kaynakta da pozisyondan (`A, B, C, ...`) üretiliyor, kaynaktan alınmıyor.
3. `tests/test_benchmarks.py` (9 test): determinizm (iki çağrı → aynı `item_id` listesi),
   `N_ITEMS` sayısı, `answer_idx` geçerli aralıkta, MMLU 57 konunun hepsi temsil ediliyor,
   MMLU `item_id`'leri tekil, ARC `item_id`'leri `id`'ye göre artan sıralı.
4. Kabul kriteri çalıştırıldı: `pytest tests/test_benchmarks.py -q` → 9 passed;
   `load_items('mmlu')` iki kez çağrıldı → `1000 True`. `pytest tests/ -q` → 25 passed.

## Önceki oturum (2026-07-25, açık bulgu temizliği + protokol geliştirme)

1. `tests/test_determinism.py` gerçek pytest testine çevrildi: `pytest.fixture(scope="module")`
   ile modeli/veri kümesini bir kez kurup iki test fonksiyonuna (`test_repeated_inference_is_bit_identical`,
   `test_injected_noise_is_detected`) böldü. Sözleşme ve mutasyon kanıtı aynı kaldı, sadece
   gerçek bir pass/fail sinyali kazandı. `pytest tests/ -v` → 16/16 passed.
2. `CLAUDE.md` Adım 2'ye AÇIK BULGULAR görev-numarası taraması eklendi — bir önceki oturumda
   tespit edilen protokol boşluğu (görev başlarken yalnızca "Ön koşul" satırı kontrol
   ediliyordu, o göreve etiketli açık bulgular otomatik taranmıyordu) kapatıldı.

## Önceki oturum (2026-07-25, Görev 2 onayı + Görev 3)

1. Görev 2 testleri (13 test, kabul kriteri çıktısı: 13/13 `NotImplementedError` ile FAIL)
   kullanıcıya sunuldu ve **onaylandı**.
2. `requirements.txt`'e `statsmodels==0.14.6` + `scipy==1.18.0` eklendi (AÇIK BULGU çözüldü),
   `requirements.lock.txt` yeniden üretildi (`pip freeze`, 56→59 paket, `patsy` transitive).
   GLM `offset=` fit'i **gerçekten çağrılarak** doğrulandı (PROTOKOL Kural 6, sadece import
   değil) — scipy 1.18'de kardeş çalışmadaki `_lazywhere` sorunu yok.
3. `src/metrics.py` implemente edildi: `ece` (equal-mass, `np.array_split`), `cal_slope`
   (`sm.GLM` + `add_constant`), `cal_intercept` (`sm.GLM` + `offset=`, sabit slope 1),
   `brier`, `overconfidence_rate`, `mean_conf_by_correctness`, `bootstrap_ci` (point =
   tam-örneklem, kwargs iletimi).
4. Kabul kriteri ilk koşuda **1 test FAIL** verdi — ama implementasyon hatası değil, testin
   kendi referans hesabında bir tuple-unpacking hatası (`expected_slope, _ = ...` yanlış
   sıradaydı). `metrics.cal_slope`'un doğruluğu referans `b1` ile ondalık düzeyinde eşleşerek
   kanıtlandı, sonra kullanıcıya sunuldu ve **onaylanan** tek satırlık düzeltme uygulandı.
   Detay: GECMIS.md "Görev 3".
5. `python -m pytest tests/ -q` → **14 passed.** `git diff HEAD -- tests/` artık boş değil
   (bilinçli, onaylı bir test-hata-düzeltmesi yüzünden) — GECMIS.md'de gerekçelendirildi.

## Önceki oturum (2026-07-25, devam)

1. `tests/test_metrics.py` yazıldı — YAPILACAKLAR Görev 2'deki 13 testin hepsi: `cal_slope`
   (mükemmel + aşırı-uçlu), `cal_intercept` (mükemmel + sistematik aşırı-güven), `ece`
   (eşit-kütle referans eşleşmesi + bin-boyutu garantisi, mükemmel kalibrasyon, dengeli+sabit
   güven ≈0.4, eşit-genişlik-kör/eşit-kütle-gören kurgu), `overconfidence_rate` (elle vaka),
   `brier` (elle vaka), `bootstrap_ci` (nokta=tam-örneklem + kwargs iletimi), clipping (0/1 güven)
2. Her sentetik veri, numpy-only Newton-Raphson referans implementasyonlarla **çalıştırılarak**
   doğrulandı (statsmodels kurulu değil) — script committe edilmedi (bkz. GECMIS.md)
3. Kabul kriteri çalıştırıldı: `ModuleNotFoundError: No module named 'src.metrics'`
   (collection hatası, tek tek `NotImplementedError` FAIL'i değil)
4. **İki açık bulgu kaydedildi:** (a) Görev 2'nin kabul kriteri metni ile "YAZMA" talimatı
   çelişiyordu — **çözüldü** (aşağıya bakınız); (b) `requirements.txt`'te `statsmodels`/`scipy`
   yok, Görev 3'ü hâlâ engelliyor (açık kaldı)
5. `bootstrap_ci` dönüş tipi `(lo, point, hi)` tuple olarak test dosyasında sabitlendi (SPEC
   bunu belirtmiyordu) — GECMIS.md'ye karar olarak yazıldı
6. **Kullanıcıyla birlikte çözüm kararlaştırıldı:** (a) bulgusu, gövdesiz bir imza iskeleti
   (`src/metrics.py`, her fonksiyon `raise NotImplementedError`, sıfır mantık) ile çözüldü —
   bu desen PROTOKOL Kural 3'e kalıcı ek olarak yazıldı ("toplu test-önce görevlerde imza
   iskeleti"), YAPILACAKLAR Görev 2'nin "Yap" talimatı buna göre güncellendi. Kabul kriteri
   şimdi tam istenen çıktıyı veriyor: 13/13 `NotImplementedError` ile FAIL.

## Önceki oturum (2026-07-25, sabah)

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

- `tests/test_no_leakage.py` henüz yok (Görev 6) — mutasyon kanıtı zorunlu (PROTOKOL Kural 4)
- Uygunluk kapısı (bf16 ≥ %50) sırası `runner.py`'de mekanik olarak zorlanmalı (Görev 7)

## Bütçe hatırlatması

Tam ızgara ≈ 140 hücre (5 model × 14 koşul × 2 benchmark). Tahmini **4–6 saat**, resumable.
Disk: `convert → evaluate → delete`, HF cache ≈ 20 GB.

**Pilotta tek hücre süresini ölç, 140 ile çarp. 8 saati aşarsa DUR ve bildir.**

---

## Oturum günlüğü

| Tarih | Ne yapıldı | Commit |
|---|---|---|
| 2026-07-24 | Fizibilite kapısı, eksen keşfi, ön-kayıt, determinizm | `ffa07c7` … `c5ea71c` |
| 2026-07-25 | Görev 1 — ortam iskeleti, config, prompt dondurma | `2a4d0c5` |
| 2026-07-25 | Görev 2 — metrik testleri yazıldı, implementasyon YOK, kullanıcı onayı bekleniyor | (önceki oturum) |
| 2026-07-25 | Görev 2 onaylandı; Görev 3 — `src/metrics.py` implementasyonu, statsmodels/scipy eklendi, test unpacking hatası düzeltildi, 14/14 yeşil | (önceki oturum) |
| 2026-07-25 | `test_determinism.py` gerçek pytest testine çevrildi (16/16 yeşil); `CLAUDE.md` protokol boşluğu kapatıldı | (önceki oturum) |
| 2026-07-25 | Görev 4 — `src/benchmarks.py` + `tests/test_benchmarks.py` (9 test), MMLU/ARC seçenek sayısı açık bulgusu çözüldü, 25/25 yeşil | (önceki oturum) |
| 2026-07-25 | Görev 5 — `src/quantize.py` + `src/measure.py`, `mixed_*` recipe açık bulgusu çözüldü, dört mod fonksiyonel doğrulandı | (bu oturum) |
| | | |

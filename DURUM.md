# DURUM — Canlı Durum Dosyası

> **Her oturum sonunda güncellenir.** Bu dosya, "neredeyiz" sorusunun tek cevabıdır.
> Son güncelleme: 2026-07-26

---

## Faz

```
[x] FAZ 0 — Fizibilite kapısı           TAMAMLANDI
[x] FAZ 1 — Keşif (eksenler, veri, hız) TAMAMLANDI
[x] FAZ 2 — Ön-kayıt donduruldu         TAMAMLANDI  (commit c5ea71c)
[x] FAZ 3 — Implementasyon              TAMAMLANDI
[x] FAZ 4 — Pilot + tam koşu            TAMAMLANDI
[ ] FAZ 5 — Analiz (ön-kayıtlı tablolar) ← ŞU AN BURADAYIZ (sırada Görev 10)
[ ] FAZ 6 — Makale + dağıtım
```

## Şu an

**FAZ 4 tamamlandı. Sırada Görev 10 (`src/analyze.py`) var.**

Görev 8 (`make pilot`) koştu, kritik açık bulgu (mkdtemp/convert path çakışması) bulundu ve
çözüldü, sağlık kontrolü kullanıcıya gösterildi ve **onaylandı**: 2-bit hücresi bf16'dan belirgin
kötü (acc 0.244 vs 0.765, ECE 0.498 vs 0.130) — kuantizasyon gerçekten uygulanıyor. Süre bütçesi
(~7.75 saat tahmini, en hızlı modelle) kullanıcıya bildirildi, kullanıcı tam koşuya onay verdi.

Görev 9 iki koşuda tamamlandı. **İlk koşu:** `caffeinate -i make reproduce` — kesinti/devam testi
geçti (~5 dk sonra kasten kesildi, yeniden başlatıldığında tamamlanmış hücrelerin `mtime`'ı
değişmedi), grid 88/88 hücre `status="ok"` verdi (140 değil — `llama3.2-1b`/`llama3.2-3b` o an
bozuk bir ölçümden `eligible=false` çıkmıştı, sadece bf16'ları çalıştı; son adım `src.analyze`
beklendiği gibi `ModuleNotFoundError` ile bitti, Görev 10 henüz yok, veri kaybı yok).

Koşu sürerken canlı izlerken **kritik bir hata bulundu**: `measure.py`'nin seçenek-harfi token'ını
`encode(...)[0]` yerine `encode(...)[-1]` ile okuması gerekiyordu — Llama-3.2 tokenizer'ı her
`encode()`'a otomatik BOS ekliyor, `[0]` hep aynı BOS token'ını okuyordu, model hep "A" seçiyormuş
gibi görünüyordu. Test-önce + mutasyon kanıtıyla, kullanıcı onaylı düzeltildi
(`tests/test_measure.py`, `pytest tests/` → 35 passed). Detay: YAPILACAKLAR AÇIK BULGULAR + GECMIS.md.

**İkinci (düzeltme) koşu:** ilk koşu tamamen bittikten sonra 4 eski/bozuk llama bf16 dosyası +
`eligibility.json` silindi, `python -m src.runner` tekrar çalıştırıldı — Qwen'in 84 hücresi
cache'ten atlandı, yalnızca 4 llama bf16 hücresi düzeltilmiş kodla yeniden ölçüldü. **Nihai sonuç:
114/114 hücre `status="ok"`, hiç `failed` yok.** `llama3.2-3b` düzeltilmiş ölçümle eşiği geçti
(arc=0.713/mmlu=0.569, eligible=true — bozuk veride bu model tamamen elenmiş olacaktı) ve tam
14-koşullu merdiveni işlendi (28 hücre); `llama3.2-1b` eşiğin altında kaldı (arc=0.475/mmlu=0.435,
yalnızca bf16, 2 hücre). Model başına hücre sayısı: `qwen2.5-0.5b` 28, `qwen2.5-1.5b` 28,
`qwen2.5-3b` 28, `llama3.2-3b` 28, `llama3.2-1b` 2.

Bu, ön-kayıtta öngörülen "havuz 4'ün altına düşebilir" riskinin gerçekleşmediğini kanıtlıyor: ana
model havuzu `qwen2.5-1.5b`, `qwen2.5-3b`, `llama3.2-3b` (3 model, eligible) + `qwen2.5-0.5b`
(floor control) — 4 model, tasarlanan gibi.
`src/runner.py::run_all` tamamen implemente edildi (üç fazlı sıra, parquet cache, teardown,
never-raise), artı Görev 7 sırasında keşfedilen bir açık bulgu (`make pilot`'ın hiç tanımlanmamış
bir CLI/model beklemesi) çözülürken eklenen `run_pilot()` + `__main__`/`argparse` girişi.
`pytest tests/ -v` → **33 passed** (regresyon yok).

Repoda şu an olanlar: `src/config.py` (artık `PILOT_MODEL`/`PILOT_CONDITIONS`/`PILOT_BENCHMARK`
dahil), `src/metrics.py` (Görev 3), `src/benchmarks.py` (Görev 4), `src/quantize.py` +
`src/measure.py` (Görev 5, tam implementasyon, dört mod da — bf16, affine, mxfp4, recipe —
gerçek dönüşümle fonksiyonel doğrulandı), `src/runner.py` (Görev 7, **tam** — `cell_id`,
`compute_eligibility`, `assert_phase3_allowed`, `run_all`, `run_pilot`, CLI), `prompts/mc_letter.txt`
(donmuş, SHA-256 `config.PROMPT_SHA256` ile korunuyor), `tests/test_prompt_frozen.py`,
`tests/test_determinism.py` (PREREG §4.6.6, gerçek `assert`'li 2 test), `tests/test_metrics.py`
(13 test), `tests/test_benchmarks.py` (9 test), `tests/test_no_leakage.py` (8 test, Görev 6),
`requirements.txt` + `requirements.lock.txt` (`statsmodels`, `scipy` dahil), `Makefile`,
`.gitignore`, `results/{cells,meta,tables,figures}` iskeleti (henüz boş — Görev 8'de ilk gerçek
hücreler yazılacak), `scratch/` (14 keşif script'i, hâlâ commit'te). `CLAUDE.md` Adım 2'deki
görev-numarası taraması bu oturumda **dördüncü kez** işe yaradı: `Makefile`'ın `pilot:` hedefi
(Görev 1'de yazılmış) hiçbir zaman implemente edilmemiş bir CLI/`--pilot` bayrağı ve adlandırılmamış
bir "bir model" bekliyordu — açık bulgu olarak kaydedilip aynı oturumda çözüldü (SPEC §9
Changelog + GECMIS.md "Görev 7").

**DUR noktası yok şu an** — Görev 7 kullanıcı onayı gerektirmiyordu (kritik koruma testi değil,
Görev 6'nın mutasyonla kanıtlanmış testlerine karşı yazılan gerçek implementasyon). Kabul kriteri
(`inspect.getsource` kontrolü) çalıştırıldı ve gösterildi; ek olarak PROTOKOL Kural 6 gereği
cache'teki 0.5B modeliyle gerçek bir uçtan uca çağrı (`scratchpad/`, `results/`'a dokunmadan)
yapılıp GECMIS.md'ye kaydedildi. Sıradaki iş Görev 8: `make pilot` çalıştırılacak ve **elle**
doğrulanacak (SPEC §8) — bu, delege edilemeyen, kullanıcının kendi gözüyle bakması gereken bir
adım.

## Son oturumda ne oldu (2026-07-25, Görev 7 — runner.py::run_all + pilot açık bulgusu)

1. `run_all(force=False)`: PHASE 1 (tüm bf16 referans hücreleri) → PHASE 2 (diskteki bf16
   parquet'lerinden `eligibility.json` yazımı, bellek içi Faz-1 sonuçlarına değil — resumability
   için) → PHASE 3 (uygun modeller + taban kontrolü için tüm kuantize hücreler). Tek yardımcı
   fonksiyon `_run_and_write_cell`: build → measure → parquet+meta yaz → teardown, tamamı
   `try/except Exception` içinde (asla `raise` etmiyor); zaten tamamlanmış hücreler (parquet
   var) `force=False` iken atlanıyor.
2. **Açık bulgu / çözüm:** `Makefile`'ın `pilot:` hedefi `python -m src.runner --pilot`
   çağırıyordu ama ne bir CLI ne de pilot modeli hiçbir yerde tanımlıydı. Aynı oturumda çözüldü:
   `config.PILOT_MODEL="qwen2.5-1.5b"` (en hızlı ana model, keşifte %73 doğruluk) +
   `run_pilot()` (aynı üç-fazlı sırayı `run_all` ile aynı yardımcı fonksiyonları kullanarak,
   `run_all`'ın SPEC imzasını değiştirmeden, SPEC §8'deki 3 hücreye uyguluyor) + `__main__`/
   `argparse --pilot`. Detay: GECMIS.md "Görev 7".
3. **Fonksiyonel doğrulama (Kural 6):** `inspect.getsource` tabanlı kabul kriteri statik; ek
   olarak `_run_and_write_cell` cache'teki `qwen2.5-0.5b` (taban kontrolü, dönüştürme
   gerektirmiyor) ile gerçekten çağrıldı — 1000 satırlık doğru şema, cache-atlama (ikinci
   çağrıda meta `mtime` değişmedi), gerçek `compute_eligibility` çıktısı
   (`{"arc_challenge": 0.518, "eligible": True, "role": "floor_control"}`), `assert_phase3_allowed`
   `raise` etmedi. Not: 0.518, feasibility'nin 30-item'lık %33 tahmininden farklı — kod hatası
   değil, örneklem büyüklüğü farkı (taban kontrolü zaten uygunluktan bağımsız tam merdiveni
   koşuyor, bu sayı Faz 3'ü etkilemiyor).
4. Kabul kriteri çalıştırıldı ve gösterildi (aşağıda). `pytest tests/ -q` → 33 passed, regresyon
   yok.

## Önceki oturum (2026-07-25, Görev 6 — test_no_leakage.py + mutasyon kanıtı)

1. **Adım 2 taraması / görev tanımı çelişkisi:** SPEC §7 build order `test_no_leakage.py`'yi
   (madde 6) `runner.py`'den (madde 7) önce sıralıyor, ama testin "eligibility.json yokken
   Faz 3 çalışamaz" ve "uygunluk yalnızca bf16'dan" sözleşmeleri `runner.py`'de bir şeye karşı
   test edilmeyi gerektiriyordu. AÇIK BULGULAR'a eklendi, aynı oturumda çözüldü: `src/runner.py`'ye
   iki saf fonksiyon (`compute_eligibility`, `assert_phase3_allowed`) SPEC §3'e eklenerek gerçek
   implemente edildi (kuantizasyon parametresi riski yok); `cell_id` de yazıldı (trivial string
   formatting); `run_all` Görev 7'de kalıyor (`NotImplementedError` iskelet). SPEC §9 Changelog.
2. `tests/test_no_leakage.py` yazıldı (8 test): 3'ü `measure.run_cell`'in warmup'ı sızdırmadığını
   (satır sayısı == item sayısı, tekrar eden `item_id` yok, çıktı id'leri girdiyle birebir),
   3'ü `runner.compute_eligibility`'nin yalnızca `condition=="bf16"` satırlarından hesapladığını
   (aynı modelin kuantize satırları karışsa bile doğruluk kirlenmiyor, floor control etiketleniyor),
   2'si `runner.assert_phase3_allowed`'ın `eligibility.json` yokken `RuntimeError` fırlattığını
   doğruluyor. Hepsi gerçek koda karşı, tek çağrıyla `pytest tests/test_no_leakage.py -v` →
   8/8 passed.
3. **Üç ayrı mutasyon kanıtı (PROTOKOL Kural 4, atlanamaz):** (a) `measure.run_cell`'e warmup
   satırlarını da çıktıya ekleyen bozukluk → 2 test `45 == 25` / tekrar eden `t0..t19` ile FAIL;
   (b) `compute_eligibility`'den bf16 filtresi kaldırıldı → beklenen `0.3` yerine kirlenmiş
   `0.6316` ile FAIL; (c) `assert_phase3_allowed`'ın kapısı `if False` ile devre dışı bırakıldı →
   `DID NOT RAISE RuntimeError` ile FAIL. Üçü de kendi doğru sebebiyle patladı, sonra dosyalar
   satır satır orijinaline döndürüldü (`git diff HEAD -- src/measure.py` boş; `runner.py` henüz
   commit'te değil, elle karşılaştırıldı). Detay: GECMIS.md "Görev 6".
4. Kabul kriteri: mutasyon çıktıları yukarıda özetlendiği gibi kullanıcıya gösterildi.
   `pytest tests/ -q` → **33 passed** (25 önceki + 8 yeni, regresyon yok).

## Önceki oturum (2026-07-25, Görev 5 — quantize.py + measure.py)

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

- Yok. Görev 8 ve Görev 9 kullanıcı onayıyla kapatıldı, Llama BOS hatası düzeltildi ve nihai
  grid (114/114 hücre `status="ok"`) doğrulandı. Sırada Görev 10 — `src/analyze.py` yazılacak.

## Bütçe hatırlatması (gerçekleşen, referans için)

Ön-kayıt tahmini ≈ 140 hücreydi (5 model × 14 koşul × 2 benchmark, tümü uygun varsayılarak).
Gerçekte `llama3.2-1b` eşiği geçemedi, bu yüzden nihai grid **114 hücre** oldu (5 model, ama
`llama3.2-1b` yalnızca bf16). Toplam koşum süresi (ilk koşu + kesinti testi + ikinci/düzeltme koşusu birlikte) 20:43 – 03:20
arası, **~6.6 saat** — pilotun en hızlı modelle yaptığı 7.75 saatlik tahminin altında kaldı
(3B modeller pilotun kullandığı 1.5B'den daha yavaş olsa da, floor control 0.5B ortalamayı
aşağı çekti ve `llama3.2-1b` yalnızca 2 hücrede kaldı, tam merdiven işlemedi).
Disk: `convert → evaluate → delete`, HF cache ≈ 20 GB.

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
| 2026-07-25 | Görev 5 — `src/quantize.py` + `src/measure.py`, `mixed_*` recipe açık bulgusu çözüldü, dört mod fonksiyonel doğrulandı | (önceki oturum) |
| 2026-07-25 | Görev 6 — `tests/test_no_leakage.py` (8 test) + `src/runner.py` kısmi implementasyonu, 3 mutasyon kanıtı | (önceki oturum) |
| 2026-07-25 | Görev 7 — `src/runner.py::run_all` + `run_pilot`/CLI, `make pilot` açık bulgusu çözüldü | (bu oturum) |
| 2026-07-25 | Görev 8 — `make pilot` koştu, mkdtemp/convert açık bulgusu çözüldü, sağlık kontrolü onaylandı | (önceki oturum) |
| 2026-07-25→26 | Görev 9 — tam koşu (kesinti/devam testi geçti), Llama BOS token açık bulgusu bulundu ve çözüldü, düzeltme koşusu — nihai 114/114 `status="ok"` | (bu oturum) |
| | | |

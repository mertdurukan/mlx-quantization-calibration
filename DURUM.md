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

**FAZ 3, Görev 1-2-3 tamamlandı. Sırada Görev 4 (`src/benchmarks.py` + örnekleme
determinizmi).** `src/metrics.py` tam olarak implemente edildi (`ece`, `cal_slope`,
`cal_intercept`, `brier`, `overconfidence_rate`, `mean_conf_by_correctness`,
`bootstrap_ci`) — `statsmodels` GLM (`offset=` ile) ve numpy kullanıyor. `tests/test_determinism.py`
gerçek pytest testine çevrildi (eski açık bulgu kapandı). `pytest tests/ -v` →
**16 passed** (13 metrik testi + 2 determinizm/mutasyon testi + 1 prompt-freeze testi).

Repoda şu an olanlar: `src/config.py`, `src/metrics.py` (tam implementasyon, Görev 3),
`prompts/mc_letter.txt` (donmuş, SHA-256 `config.PROMPT_SHA256` ile korunuyor),
`tests/test_prompt_frozen.py`, `tests/test_determinism.py` (PREREG §4.6.6, artık gerçek
`assert`'li 2 test — `test_repeated_inference_is_bit_identical`,
`test_injected_noise_is_detected`), `tests/test_metrics.py` (13 test, hepsi geçiyor — bir
unpacking hatası kullanıcı onayıyla düzeltildi, bkz. aşağı), `requirements.txt` +
`requirements.lock.txt` (artık `statsmodels`, `scipy` dahil), `Makefile`, `.gitignore`,
`results/{cells,meta,tables,figures}` iskeleti, `scratch/` (14 keşif script'i, hâlâ commit'te).
`CLAUDE.md` Adım 2'ye bir kural eklendi: görev başlamadan önce `AÇIK BULGULAR`'da o görev
numarasına etiketli işaretlenmemiş bulgu var mı taranacak (önceki oturumda kullanıcıya
sorulmuş, bu oturumda onaylanıp uygulandı).

**DUR noktası yok şu an** — Görev 3 kullanıcı onayı gerektirmiyordu (test onayı Görev 2'de
alınmıştı). Sıradaki iş Görev 4.

## Son oturumda ne oldu (2026-07-25, açık bulgu temizliği + protokol geliştirme)

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
| 2026-07-25 | Görev 1 — ortam iskeleti, config, prompt dondurma | `2a4d0c5` |
| 2026-07-25 | Görev 2 — metrik testleri yazıldı, implementasyon YOK, kullanıcı onayı bekleniyor | (önceki oturum) |
| 2026-07-25 | Görev 2 onaylandı; Görev 3 — `src/metrics.py` implementasyonu, statsmodels/scipy eklendi, test unpacking hatası düzeltildi, 14/14 yeşil | (önceki oturum) |
| 2026-07-25 | `test_determinism.py` gerçek pytest testine çevrildi (16/16 yeşil); `CLAUDE.md` protokol boşluğu kapatıldı | (bu oturum) |
| | | |

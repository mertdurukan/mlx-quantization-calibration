# YAPILACAKLAR

> Sıralı. **İşaretlenmemiş ilk görev** her zaman sıradaki görevdir.
> Her görevin kabul kriteri **çalıştırılabilir bir komuttur** — çıktısı kullanıcıya gösterilir.

---

## FAZ 3 — Implementasyon

### [x] Görev 1 — Ortam iskeleti + prompt şablonunu dondur — TAMAMLANDI 2026-07-25
**Ön koşul:** yok
**Yap:**
- `src/`, `tests/`, `prompts/`, `results/{cells,meta,tables,figures}`, `scratch/` oluştur
- `src/config.py`'yi SPEC §2'den **birebir** yaz (tek gerçek kaynak)
- `prompts/mc_letter.txt` yaz — tek şablon, PREREG §4.4'teki format
- `requirements.txt` + `requirements.lock.txt` (`pip freeze`) — sabit sürümler
- `Makefile` — SPEC §6'daki hedefler, hepsi `./.venv/bin/python` çağırsın
- `.gitignore` — `.venv/`, `scratch/`, `results/cells/`, `results/meta/`, `__pycache__/`
- `tests/test_prompt_frozen.py` — şablonun SHA-256'sını `config.PROMPT_SHA256` ile karşılaştırır
- Keşif script'lerini `scratch/`'e taşı (silme — fizibilite kanıtı)

**Kabul kriteri:**
```bash
make test 2>&1 | tail -3 && python -c "import src.config as c; print(c.N_ITEMS, len(c.CONDITIONS))"
```
→ prompt testi geçmeli, `1000 14` yazmalı

---

### [x] Görev 2 — Metrik testleri (TESTLER ÖNCE, implementasyon YOK) — TAMAMLANDI 2026-07-25
**Ön koşul:** Görev 1
**Yap:** `tests/test_metrics.py` — bilinen-cevap testleri. `src/metrics.py`'ye **implementasyon
YAZMA** — yalnızca PROTOKOL Kural 3 ekindeki imza iskeleti (her gövde `raise
NotImplementedError`, sıfır mantık) serbest, gövde doldurma Görev 3'te.

Testler:
- `cal_slope`: mükemmel kalibre sentetik veride 1.0 ± 0.05
- `cal_slope`: kasten aşırı-uçlu tahminlerde belirgin < 1
- `cal_intercept`: mükemmel veride 0.0 ± 0.05; sistematik aşırı-güvende belirgin < 0
- `ece`: eşit-KÜTLE binleme — bin sayıları en fazla 1 fark etmeli
- `ece`: mükemmel kalibre → < 0.01
- `ece`: dengeli etiket + sabit 0.9 güven → ≈ 0.4 (⚠️ etiketleri **seed'li shuffle** et,
  sıralı bırakırsan doğru kodda bile patlar — kardeş çalışmada tam bu hata oldu)
- `ece`: eşit-genişlik binlemenin **kör kaldığı**, eşit-kütlenin **gördüğü** bir kurgu
- `overconfidence_rate`: elle hesaplanmış 10 örnekli vaka, tam eşleşme
- `brier`: elle hesaplanmış vaka
- `bootstrap_ci`: `lo < point < hi`, ve **point = tam örneklem metriği** (bootstrap ortalaması
  DEĞİL)
- `bootstrap_ci`: **kwargs iletimi** — `bootstrap_ci(..., overconfidence_rate, threshold=0.9)`
  çalışmalı (kardeş çalışmada bu bug analiz aşamasında çökerdi)
- clipping: güven tam 0.0 ve 1.0 içerdiğinde `cal_slope`/`cal_intercept` sonlu dönmeli

**Kabul kriteri:**
```bash
python -m pytest tests/test_metrics.py -q 2>&1 | tail -5
```
→ **hepsi `NotImplementedError` ile FAIL etmeli** (test kodunun kendisinde hata olmamalı)

**DUR.** Testleri kullanıcıya göster, onay al. Sonra Görev 3.

---

### [x] Görev 3 — `src/metrics.py` implementasyonu — TAMAMLANDI 2026-07-25
**Ön koşul:** Görev 2 onaylandı
**Yap:** Testleri geçir. **Hiçbir testi değiştirme, gevşetme, silme.** Test yanlışsa DUR ve söyle.
- `cal_intercept` → `statsmodels` GLM, `offset=` ile, eğim 1'e sabit
- `ece` → 15 eşit-kütle bin, `np.array_split`
- `bootstrap_ci` → `**metric_kwargs` hem nokta tahminine hem her resample'a iletilir

**Kabul kriteri:**
```bash
git diff HEAD -- tests/ && python -m pytest tests/ -q 2>&1 | tail -3
```
→ `git diff` **BOŞ**, testlerin hepsi geçmeli

---

### [x] Görev 4 — `src/benchmarks.py` + örnekleme determinizmi — TAMAMLANDI 2026-07-25
**Ön koşul:** Görev 3
**Yap:** SPEC §3'teki `Item` ve `load_items`.
- ARC-Challenge: `id`'ye göre sıralı ilk 1000
- MMLU: 57 konuya orantılı stratified, seed 0, artık konu adına göre dağıtılır
- Harfler **pozisyondan** üretilir (A, B, C…), kaynaktan alınmaz — iki benchmark aynı skorlanır
- `tests/test_benchmarks.py`: aynı çağrı iki kez → **birebir aynı item_id listesi**;
  MMLU'da 57 konunun hepsi temsil edilmiş; her Item'da `answer_idx` geçerli aralıkta

**Kabul kriteri:**
```bash
python -m pytest tests/test_benchmarks.py -q 2>&1 | tail -3 && python -c "
from src.benchmarks import load_items
a=load_items('mmlu'); b=load_items('mmlu')
print(len(a), [x.item_id for x in a[:3]]==[x.item_id for x in b[:3]])"
```
→ testler geçmeli, `1000 True`

---

### [x] Görev 5 — `src/quantize.py` + `src/measure.py` — TAMAMLANDI 2026-07-25
**Ön koşul:** Görev 4
**Yap:** SPEC §3'teki sözleşmeler.
- `quantize.build` → `snapshot_download` **önce**, sonra `convert`; `effective_bits` **dönüşüm
  logundan parse edilir**, varsayılmaz
- `quantize.teardown` → dönüştürülmüş dizini siler, HF cache'e dokunmaz
- `measure.run_cell` → ilk 20 item warmup, **atılır**; güven = seçenek-harf token'larının
  softmax'ı; tek item hatası hücreyi düşürmez

**Kabul kriteri:**
```bash
python -c "
from src.quantize import build, teardown
r = build('qwen2.5-1.5b','affine_b4_g64','/tmp/q_t1')
print('effective_bits:', r['effective_bits'], '| nominal 4 ile ayni mi:', r['effective_bits']==4.0)
teardown('/tmp/q_t1')"
```
→ `effective_bits` ≈ 4.50 olmalı ve `False` yazmalı (nominal ≠ efektif)

---

### [x] Görev 6 — Sızıntı/sıra sözleşme testleri + **MUTASYON KANITI** — TAMAMLANDI 2026-07-25
**Ön koşul:** Görev 5
**Yap:** `tests/test_no_leakage.py`:
- Warmup item'ları çıktıda **yok** (n_scored == n_items, warmup ayrı sayılır)
- FAZ 3 (kuantize hücreler) `results/eligibility.json` **yokken çalışamaz**
- Uygunluk kararı **yalnızca bf16 satırlarından** hesaplanıyor

**Sonra MUTASYON (PROTOKOL Kural 4, atlanamaz):**
1. Kasten bozuk bir runner yaz: warmup item'larını da skorlayan / eligibility'yi kuantize
   sonuçlardan hesaplayan
2. Testi ona karşı koştur → **doğru assertion'da patlamalı**
3. Bozuk kodu sil

**Kabul kriteri:** Mutasyon çıktısı kullanıcıya gösterilir: testin bozuk implementasyonu
**reddettiği** görülmeli. Görmeden bu görev kapanmaz.

---

### [x] Görev 7 — `src/runner.py` — TAMAMLANDI 2026-07-25
**Ön koşul:** Görev 6 (mutasyon kanıtı dahil)
**Yap:** SPEC §3'teki `run_all`, **zorunlu üç faz sırasıyla**:
FAZ 1 tüm bf16 → FAZ 2 `eligibility.json` yaz → FAZ 3 kuantize hücreler.
- Hücre başına parquet cache, tamamlanmışı atla
- Her non-bf16 hücreden sonra `teardown`
- Asla `raise` etme, asla atlama; hata `status="failed"` + exception metni

**Kabul kriteri:**
```bash
python -c "
import inspect, src.runner as r
s = inspect.getsource(r)
for k in ['eligibility','teardown','exists','status']: print(k, k in s)"
```
→ hepsi `True`

---

## FAZ 4 — Pilot + tam koşu

### [x] Görev 8 — `make pilot` + **ELLE** doğrulama — TAMAMLANDI 2026-07-25
**Ön koşul:** Görev 7
**Yap:** 1 model × 3 koşul (bf16, affine_b4_g64, affine_b2_g64) × arc_challenge

**Kabul kriteri — bunu delege etme, kullanıcı kendi gözüyle bakar:**
- 3 hücre, `status="ok"`, her birinde 1000 skorlanmış + 20 warmup atılmış
- `effective_bits` var ve nominalden farklı
- ⚠️ **2-bit hücresi bf16'dan belirgin KÖTÜ** olmalı (hem doğruluk hem ECE).
  Aynıysa kuantizasyon **uygulanmıyor** demektir — keşifte tam bu tuzağa düşüldü
  (`nn.quantize` çıplak `Linear`'da sessizce no-op).
- Tek hücre süresini ölç × 140. **8 saati aşarsa DUR ve bildir.**

---

### [ ] Görev 9 — Tam koşu
**Ön koşul:** Görev 8 elle onaylandı
**Yap:** `caffeinate -i make reproduce`. Kesintiye dayanıklı olmalı — başladıktan 2 dk sonra
`Ctrl+C` yapıp tekrar başlat, tamamlanmış hücreleri atladığını **doğrula**.

**Kabul kriteri:** 140 hücre, `status` dağılımı, her farklı hatanın **tam metni**.
Başarısız hücre varsa `DEVIATIONS.md`'ye açıklaması yazılır — **silinmez.**

**Sonuçlara bakma.** Bu görevin çıktısı sağlık raporudur, metrik ortalaması değil.

---

## FAZ 5 — Analiz

### [ ] Görev 10 — `src/analyze.py` — sadece ön-kayıtlı tablolar
**Ön koşul:** Görev 9
**Yap:** PREREG §5'teki **tam olarak** 4 tablo + 3 figür. Fazlası "Exploratory" etiketiyle
ayrı bölümde.
- Tablo 1 (H1): bit merdiveni, bf16'ya eşleştirilmiş, monotonluk hükmü
- Tablo 2 (H2): doğru/yanlış güven ayrımı, aşırı-güven oranı, yön hükmü
- Tablo 3 (H3): affine vs mxfp4 @ 4-bit/g32
- Tablo 4 (H4): mixed recipe'ler, bileşen bit'leriyle
- Figür 1: kalibrasyon eğrileri · Figür 2: ECE ~ **efektif** bit · Figür 3: güven dağılımı
- Her sayı %95 aralıklı. CV yok, bootstrap item üzerinden.

**Kabul kriteri:** Her tablo bir hipoteze 1:1 karşılık gelir, PASS/FAIL hükmü var, ana
tablolarda ön-kayıtsız hiçbir şey yok.

---

### [ ] Görev 11 — Veri tarama yasağı kontrolü
**Ön koşul:** Görev 10
**Yap:** Analiz çıktısını gözden geçir: ön-kayıtta olmayan hiçbir test yapılmamış mı?
Yapıldıysa "Exploratory" etiketi var mı? Yoksa kaldır veya etiketle.

---

## FAZ 6 — Yazı + dağıtım

### [ ] Görev 12 — `README.md`
Repo ön yüzü: tek paragraf bulgu, manşet sayılar, `make reproduce`, sınırlar, abartı yok.

### [ ] Görev 13 — `paper.md`
Abstract, giriş, related work (**kaynaklar tam metin okunacak**, ≤15 kelime alıntı), yöntem,
sonuçlar (gömülü tablo+figür), tartışma, sınırlar, reprodüksiyon.

### [ ] Görev 14 — Dağıtım
GitHub vitrini (description, topics), release + Zenodo DOI, sonra arXiv / doğrudan e-posta.

---

## 🔍 AÇIK BULGULAR

> Çalışırken keşfedilen her eksik, belirsizlik, tutarsızlık **buraya yazılır** — sessizce
> geçilmez. Format:
> `- [ ] **[TARİH]** <bulgu> · Engellediği görev: <no|yok> · Ciddiyet: kritik|orta|düşük`

- [ ] **[2026-07-24]** `mxfp8` group size kısıtı doğrulanmadı (mxfp4'ün 32 olduğu kanıtlandı, mxfp8 varsayıldı). Ön-kayıtta mxfp8 koşulu yok, o yüzden şimdilik engel değil. · Engellediği görev: yok · Ciddiyet: düşük
- [x] **[2026-07-24]** `mixed_*` recipe'lerin nasıl çağrıldığı (`quant_predicate` string mi, builder fonksiyonu mu) doğrulanmadı. Görev 5'te kanıtlanmalı. **Çözüldü (2026-07-25, Görev 5 öncesi):** `mlx_lm/convert.py` kaynağı okundu + gerçek `convert()` çağrısıyla doğrulandı — `quant_predicate` doğrudan `condition_tag` string'i (`"mixed_2_6"` vb., `QUANT_RECIPES` ile birebir eşleşiyor), `q_mode="affine"` zorunlu, `q_group_size=None`/`q_bits=None` (recipe kendi per-layer bit/group değerlerini döndürüyor, gerçek group size iç varsayılan 64'e düşüyor — `config.json`'da doğrulandı). Detay: GECMIS.md. · Engellediği görev: yok (çözüldü) · Ciddiyet: orta
- [ ] **[2026-07-24]** Llama-3.2-1B ve 3B'nin doğruluğu hiç ölçülmedi (sadece erişilebilirliği doğrulandı). Uygunluk kapısı bunu FAZ 1'de zaten yakalayacak, ama havuzun 4'ten aza düşme riski var. · Engellediği görev: yok · Ciddiyet: orta
- [x] **[2026-07-24]** MMLU'da seçenek sayısı her zaman 4 mü, doğrulanmadı. `n_options` sabit varsayılmamalı. **Çözüldü (2026-07-25, Görev 4 öncesi):** `cais/mmlu` `all`/test split'i (14042 satır) **çalıştırılarak** tarandı — hepsi tam 4 seçenekli. Ayrıca `allenai/ai2_arc` ARC-Challenge test split'i (1172 satır) de tarandı: seçenek sayısı **{3, 4, 5} arasında değişiyor** — bu da SPEC'teki `Item.options: list[str]` + `n_options` kolonunun sabit varsayılmaması gerektiğini doğruluyor, `load_items`/`run_cell` seçenek sayısını asla 4'e sabitlememeli. Detay: GECMIS.md. · Engellediği görev: yok (çözüldü) · Ciddiyet: orta
- [x] **[2026-07-25]** `tests/test_determinism.py` pytest `assert` içermiyor — top-level script, import edildiğinde (yani her `make test`'te) gerçek bir 1.5B model conversion'ı çalıştırıyor ve sadece print ediyor. `pytest` onu topluyor ama hiçbir test fonksiyonu bulamıyor (0 test), yine de içeriği collection sırasında **çalışıyor** — her `make test` artık ağ/HF-cache'e bağımlı ve yavaş, ve gerçek bir pass/fail sinyali yok. PREREG §4.6.6 sözleşmesi "geçti" olarak commit edilmişti ama pytest formatında değil. **Çözüldü:** `pytest.fixture(scope="module")` + iki gerçek test fonksiyonuna (`test_repeated_inference_is_bit_identical`, `test_injected_noise_is_detected`) çevrildi; sözleşme ve mutasyon kanıtı aynı, artık gerçek pass/fail veriyor. `pytest tests/ -v` → 16/16 passed (bkz. 2 yeni test). · Engellediği görev: yok · Ciddiyet: orta
- [x] **[2026-07-25]** Görev 2 kabul kriteri metni ("hepsi `NotImplementedError` ile FAIL etmeli") ile aynı görevin "Yap" talimatı ("`src/metrics.py`'yi **YAZMA**") çelişiyordu: modül hiç yoksa import `ModuleNotFoundError` ile collection hatası verir, tek tek `NotImplementedError` FAIL'i değil. **Çözüldü:** kullanıcıyla birlikte karar verildi — `src/metrics.py`'ye SPEC §3'ün **imza iskeleti** yazıldı (her gövde yalnızca `raise NotImplementedError`, sıfır mantık). PROTOKOL Kural 3'e bu desen kalıcı ek olarak yazıldı. `pytest tests/test_metrics.py -q` artık 13/13 `NotImplementedError` ile FAIL veriyor — doğrulandı. · Engellediği görev: yok · Ciddiyet: düşük
- [x] **[2026-07-25]** `requirements.txt` içinde `statsmodels` ve `scipy` **yok**, ama SPEC §3 `cal_intercept`'in "`statsmodels` GLM, `offset=` ile" uygulanmasını zorunlu kılıyor. Şu an kurulu değiller (`ModuleNotFoundError`). Görev 2'nin referans hesaplarını (bkz. `tests/test_metrics.py`) numpy-only Newton-Raphson ile bağımsız doğruladım, bu yüzden Görev 2'yi engellemiyor — ama Görev 3 (`src/metrics.py` implementasyonu) bu bağımlılık eklenmeden başlayamaz (SPEC §0 madde 8: sabitlenmemiş bağımlılık eklenemez, `requirements.txt` + `requirements.lock.txt` aynı commit'te güncellenmeli). **Çözüldü:** `statsmodels==0.14.6` + `scipy==1.18.0` eklendi, `requirements.lock.txt` yeniden üretildi (`pip freeze`), GLM `offset=` fit'i fonksiyonel olarak doğrulandı (PROTOKOL Kural 6). · Engellediği görev: 3 · Ciddiyet: orta
- [x] **[2026-07-25]** `tests/test_metrics.py::test_cal_slope_perfect_calibration_is_near_one` içinde gerçek bir hata bulundu: `expected_slope, _ = _fit_reference_cal_slope_intercept(y, conf)` tuple'ı yanlış sırayla açıyordu (fonksiyon `(intercept, slope)` döndürüyor, test `expected_slope`'a intercept'i atıyordu). `metrics.cal_slope`'un doğru olduğu bağımsız doğrulandı (referans `b1` ile ondalık düzeyinde eşleşme: 1.0124437146123444 vs 1.012443714612344). **Çözüldü:** kullanıcı onayıyla tek satır düzeltildi (`_, expected_slope = ...`), bkz. GECMIS.md "Görev 3". · Engellediği görev: yok (çözüldü) · Ciddiyet: orta
- [x] **[2026-07-25]** Görev 6 (`tests/test_no_leakage.py`), SPEC §7'ye göre `runner.py`'den (Görev 7) önce sıralanıyor, ama testin sözleşmeleri ("eligibility.json yokken Faz 3 çalışamaz", "uygunluk yalnızca bf16'dan") `runner.py`'de bir şeye karşı test edilmeyi gerektiriyor — modül hiç yoksa test totolojik/imkansız olurdu. **Çözüldü:** `src/runner.py`'ye SPEC §3'e eklenen iki saf fonksiyonla (`compute_eligibility`, `assert_phase3_allowed`) kısmi implementasyon yazıldı (kuantizasyon parametresi seçmiyor/değiştirmiyor, SPEC §0 madde 1 riski yok); `cell_id`/`run_all` Görev 7'de kalıyor. SPEC §9 Changelog + GECMIS.md "Görev 6". · Engellediği görev: yok (çözüldü) · Ciddiyet: orta
- [x] **[2026-07-25]** `make pilot` çalıştırıldı: bf16 hücresi `status="ok"` (1000 skorlanmış, 206.7s), ama **her iki kuantize hücre de (`affine_b4_g64`, `affine_b2_g64`) `status="failed"`** — hiçbiri dönüşüme bile ulaşmadı (`n_items_scored: 0`, `wall_seconds` <0.4s). Kök neden: `runner.py:79`'daki `_run_and_write_cell`, `out_dir`'i `tempfile.mkdtemp(prefix=...)` ile **önceden oluşturuyor**, ama `mlx_lm.convert()` kaynak kodu (`Path(mlx_path).exists()` kontrolü) hedef dizin **zaten varsa** `ValueError` fırlatıyor ("Cannot save to the path ... as it already exists"). Bu, `runner.py` üzerinden çağrılan **her** non-bf16 hücrede %100 tekrarlanır — kuantizasyon parametresi sorunu değil, saf dizin-yönetimi hatası. Görev 5'in fonksiyonel doğrulaması bunu yakalamadı çünkü `quantize.build()` doğrudan, `tempfile.mkdtemp` kullanmayan sabit bir yol (`/tmp/q_t1`, önceden var olmayan) ile çağrılmıştı — hata yalnızca `runner.py`'nin orkestrasyon yolundan geçince ortaya çıkıyor. Sonuç: **hiçbir kuantize hücre şu ana kadar `runner.py` üzerinden gerçekten hiç çalışmadı**, SPEC §8 kabul kriterinin 2-bit/bf16 sağlık kontrolü hiç görülemedi. `never-raise` sözleşmesi doğru çalıştı (crash olmadı, `status="failed"` + hata metni yazıldı) ama görev tamamlanamaz durumda. **Çözüldü:** `_run_and_write_cell`, `mkdtemp` ile tekil bir isim üretip hemen `os.rmdir` ile siliyor — `mlx_lm.convert()`'e her zaman **yeni ama garantili tekil** bir yol veriliyor. `make pilot` yeniden koşuldu: 3/3 hücre `status="ok"`, `affine_b4_g64` effective_bits=4.501, `affine_b2_g64` effective_bits=2.501; doğruluk+ECE sağlık kontrolü aşağıda (bkz. DURUM.md) — 2-bit belirgin kötü, kuantizasyon gerçekten uygulanıyor. `pytest tests/ -q` → 33 passed, `git diff -- tests/` boş. · Engellediği görev: 8 (çözüldü) · Ciddiyet: kritik
- [x] **[2026-07-25]** `Makefile`'daki `pilot:` hedefi `$(PYTHON) -m src.runner --pilot` çağırıyor (Görev 1'de yazıldı), ama SPEC §3'ün `runner.py` sözleşmesi yalnızca `run_all(force: bool=False)`'ı tanımlıyor — ne bir CLI/`__main__` girişi ne de `--pilot` modu var. Ayrıca SPEC §8/YAPILACAKLAR Görev 8 "bir model" diyor ama **hangi model** hiçbir yerde adlandırılmamış. Görev 7 sırasında `run_all` yazılırken keşfedildi. **Çözüldü (aynı oturum):** `src/runner.py`'ye `run_pilot()` (SPEC §8'deki 3 hücreyi — bf16, affine_b4_g64, affine_b2_g64 × arc_challenge — aynı üç-fazlı sırayla, `_run_and_write_cell`/`compute_eligibility`/`assert_phase3_allowed` yeniden kullanılarak koşturuyor) ve `if __name__=="__main__"` + `argparse --pilot` eklendi; `run_all`'ın imzası **değişmedi** (SPEC sözleşmesi korundu). Model seçimi: `qwen2.5-1.5b` (dört ana modelin en hızlısı, keşif fazında %73 doğrulukla uygunluk barının belirgin üzerinde — bilinçli bir mühendislik kararı, PREREG'e dokunmuyor çünkü pilot verisi §0 Pilot İfşası gereği zaten nihai tablolara girmiyor). SPEC §9 Changelog + GECMIS.md "Görev 7". · Engellediği görev: 8 (çözüldü) · Ciddiyet: orta

- [ ] **[2026-07-25]** `Makefile`'daki `reproduce:` hedefi `test && runner && analyze` sırasıyla çalışıyor, ama `src/analyze.py` henüz yazılmadı (Görev 10). `caffeinate -i make reproduce` bu yüzden grid tamamlandıktan **sonra** `ModuleNotFoundError`/`No module named src.analyze` ile beklenen bir hatayla bitecek — veri kaybı yok (runner hücre başına diske zaten yazıyor), sadece `make`'in çıkış kodu 0 olmayacak. Görev 9'un kabul kriteri yalnızca grid'in status dağılımını istiyor, analyze'i değil, o yüzden engel değil. · Engellediği görev: yok · Ciddiyet: düşük
- [ ] **[2026-07-25]** Görev 9 sürerken (37/140 hücre) `eligibility.json`'da `llama3.2-1b` ve `llama3.2-3b` için **birebir aynı** sayılar görüldü (`arc_challenge: 0.222`, `mmlu: 0.238`, ikisi de) — iki farklı boyutlu modelin bire bir aynı doğruluğu vermesi kendi başına şüpheli, kontrol edildi. Kök neden bulundu ve **kanıtlandı**: `src/measure.py:26`'daki `_score_item`, seçenek harfinin logit indeksini `tokenizer.encode(" " + label)[0]` ile alıyor — yani encode çıktısının **ilk** token'ı. Qwen tokenizer'ında bu doğru (`encode(" A") == [362]`, tek token). Ama Llama-3.2 tokenizer'ı her `encode()` çağrısına otomatik `<|begin_of_text|>` (id 128000) **başa ekliyor** (`encode(" A") == [128000, 362]`) — kod dört seçenek için de **aynı sabit BOS token'ının** logit'ini okuyor, dördü de birebir eşit çıkıyor, softmax sonrası hepsi 0.25 (kontrol edildi: `llama3.2-1b__bf16__arc_challenge.parquet`'te `pred_idx` 1000/1000 satırda sabit `0`, `conf_pred` std=0.0). **Sonuç: Llama-3.2-1B/3B için ölçülen "doğruluk" model kalitesini yansıtmıyor — sabit "her zaman A seç" davranışının seçenek dağılımına denk gelen bir artefakt.** İki bf16 hücresi (`llama3.2-1b`/`llama3.2-3b` × `arc_challenge`/`mmlu`, 4 hücre) ve bunlardan türetilen `eligibility.json` girdileri geçersiz. **Qwen ailesi etkilenmiyor** (tokenizer'ları `encode()`'da BOS eklemiyor, `eligibility.json`'daki Qwen sayıları zaten çeşitli/anlamlı — pilot sonuçlarıyla tutarlı). Arka planda süren Görev 9 koşusu şu an yalnızca (uygun bulunan) Qwen kuantize hücrelerini işliyor, bu yüzden **daha fazla hesaplama israfı sürmüyor** — ama Llama'nın gerçek uygunluğu hâlâ bilinmiyor (düzeltilirse eşiği geçebilir, bu da onu tüm ızgaraya sokar). Düzeltme muhtemelen basit (`[0]` yerine `[-1]` — Llama'da son token her zaman etiketin kendisi: `[128000, 362][-1] == 362`; Qwen'de tek elemanlı listede `[-1] == [0]`, geriye dönük kırmıyor) ama `measure.py` SPEC §3 sözleşmesi altında kritik/korumalı kod, kullanıcı onayı olmadan değiştirilmedi. · Engellediği görev: 9 (Llama kapsamı) · Ciddiyet: **kritik**

---

## Tamamlananlar

- [x] **2026-07-24** FAZ 0 — Fizibilite kapısı (log-prob okunuyor, iki seviye karşılaştırılabiliyor)
- [x] **2026-07-24** FAZ 1 — Eksen/veri/hız keşfi, hepsi çalıştırılarak doğrulandı
- [x] **2026-07-24** FAZ 2 — Ön-kayıt donduruldu ve push'landı (`c5ea71c`)
- [x] **2026-07-24** Determinizm sözleşmesi (PREREG §4.6.6) — mutasyonla kanıtlandı
- [x] **2026-07-25** Görev 1 — ortam iskeleti, `src/config.py`, `prompts/mc_letter.txt` donduruldu, `make test` yeşil, `1000 14` doğrulandı
- [x] **2026-07-25** Görev 2 — `tests/test_metrics.py` (13 test) yazıldı, `src/metrics.py` imza iskeleti, kullanıcı onayı alındı
- [x] **2026-07-25** Görev 3 — `src/metrics.py` implementasyonu; `statsmodels`/`scipy` eklendi; testte bulunan unpacking hatası kullanıcı onayıyla düzeltildi; `pytest tests/` → 14 passed
- [x] **2026-07-25** Görev 4 — `src/benchmarks.py` (`Item`, `load_items`) + `tests/test_benchmarks.py` (9 test); Görev 4'ü engelleyen açık bulgu (MMLU seçenek sayısı) çözüldü; `pytest tests/` → 25 passed
- [x] **2026-07-25** Görev 5 — `src/quantize.py` (`build`, `teardown`) + `src/measure.py` (`run_cell`); Görev 5'i engelleyen açık bulgu (`mixed_*` recipe çağrı biçimi) çözüldü; dört mod da (bf16, affine, mxfp4, recipe) gerçek dönüşümle fonksiyonel doğrulandı; kabul kriteri → `effective_bits: 4.501 | ... False`
- [x] **2026-07-25** Görev 6 — `tests/test_no_leakage.py` (8 test) + `src/runner.py` kısmi implementasyonu (`cell_id`, `compute_eligibility`, `assert_phase3_allowed`); üç ayrı mutasyon (warmup sızıntısı, uygunluk sızıntısı, sıra kapısı devre dışı) üçü de doğru sebeple FAIL verdi, bozuk kod silindi; `pytest tests/` → 33 passed
- [x] **2026-07-25** Görev 7 — `src/runner.py::run_all` (üç fazlı sıra, parquet cache, teardown, never-raise) + `run_pilot()`/CLI (Görev 7'yi engelleyen açık bulgu: `make pilot`'ın hiç tanımlanmamış bir CLI/model beklediği çözüldü); `inspect`-tabanlı kabul kriteri geçti, ek olarak cache'teki 0.5B ile gerçek uçtan uca çağrı yapılıp doğrulandı (1000 satır doğru şema, cache-atlama, gerçek eligibility, faz-3 kapısı); `pytest tests/` → 33 passed (regresyon yok)
- [x] **2026-07-25** Görev 8 — `make pilot` koştu; kritik açık bulgu (mkdtemp/convert path çakışması) bulundu ve çözüldü (`mkdtemp` + hemen `os.rmdir`); 3/3 hücre `status="ok"` (bf16 16.0 bit, affine_b4_g64 4.501 bit, affine_b2_g64 2.501 bit); sağlık kontrolü kullanıcıya gösterildi ve **onaylandı**: bf16 acc=0.765/ECE=0.130, b4 acc=0.737/ECE=0.136, b2 acc=0.244/ECE=0.498 — 2-bit belirgin kötü, kuantizasyon gerçekten uygulanıyor; süre bütçesi (~199s/hücre × 140 ≈ 7.75 saat, en hızlı modelle) kullanıcıya bildirildi ve tam koşuya onay alındı; `pytest tests/` → 33 passed

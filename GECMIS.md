# GEÇMİŞ — Kararlar ve Yakalanan Hatalar

> Append-only. Her karar **gerekçesiyle**, her yakalanan hata **nasıl yakalandığıyla**.
> Bu dosya "neden böyle?" sorusunun cevabıdır. Silinmez, düzenlenmez.

---

## Neden bu konu

**Terk edilen açı (2026-01 → 2026-07):** "Kuantizasyon kalibrasyonu bozuyor mu? Üç makale
çelişiyor" (Proskurina NAACL 2024 / Zhong ACL 2025 / Singh AACL 2025). Ocak 2026 bilgisiyle
bu çelişki açık görünüyordu.

**Temmuz 2026'da doğrulandığında çelişki büyük ölçüde ÇÖZÜLMÜŞTÜ.** 2026 çalışmaları desende
birleşmiş; ayrıca ACL-SELVA 2026 çalışması GPTQ/AWQ kalibrasyon-verisi duyarlılığının farklı
mekanizmalardan geldiğini göstermiş.

**Sonuç:** Konu terk edildi, **ön-kayıt yazılmadan önce.** Üç hafta kurtarıldı.
Bu, PROTOKOL Kural 1'in (tazelik kontrolü) tek başına gerekçesi.

**Seçilen açı:** MLX kuantizasyon merdiveni ve kalibrasyon. Boşluk Temmuz 2026'da doğrulandı:
MLX literatürü throughput/perplexity/doğruluk ölçüyor, kalibrasyona **hiç** bakmıyor;
LLM-kalibrasyon literatürü kuantizasyona **hiç** bakmıyor. Kesişim boş.

---

## Verilen tasarım kararları

### Tek makine, tek runtime, tek değişken
4060'lı Windows makinesi de var (CUDA), ama **v1'de kullanılmıyor.** Farklı donanımda ölçülen
ECE'ler doğrudan kıyaslanamaz — donanım bir karıştırıcıdır. Cross-runtime karşılaştırma
ayrı bir çalışma.

### Kendi dönüştürmemiz, hazır checkpoint değil
mlx-community'nin yayınlanmış bf16 ve 4-bit sürümleri farklı zamanlarda, farklı MLX
sürümleriyle üretilmiş olabilir. Tek bf16 kaynaktan kendi dönüşümümüz = **tek değişken.**
(Kardeş ML çalışmasında GPTQ/bitsandbytes karışıklığında görülen hatanın önlenmesi.)

### Koşul 8 taşıyıcı: `affine` 4-bit @ g=32
`mxfp4` group size 32 **zorunlu** (16/64/128 açıkça reddediliyor — çalıştırılarak kanıtlandı).
Onu `affine` g=64 ile kıyaslarsak *mod* ile *group size* karışır. Koşul 8, RQ3'ün istediği
eşleşmiş karşılaştırmayı sağlar.

### Log-prob güveni, sözel güven değil
Sözel güven ("%90 eminim" dedirtmek) bağımsız olarak bozuk — yayınlanmış ECE > 0.377,
tahminler 90-100 bandında kümeleniyor. Onu ölçersek kuantizasyonun etkisi gürültüde kaybolur.

### Tek protokol: harf-tabanlı MC
HellaSwag ve PIQA elendi — devam-skorlama (uzunluk-normalize log-prob) **farklı bir ölçüm
protokolü.** Kalibrasyon protokole aşırı duyarlı; ikisini karıştırmak kuantizasyon etkisini
protokol etkisiyle karıştırır.

### 0.5B ana ızgaradan çıkarıldı
30 ARC-Challenge item'ında %33 doğruluk — şans (%25) seviyesine yakın. Kalibrasyon "emin
olduğunda haklı mı?" sorusudur; model hiçbir şeyden emin değilse ölçecek şey yok.
**Ama post-hoc dışlama değil:** ön-kayıtta "bf16 doğruluğu ≥ %50" diye **mekanik bir kural**
yazıldı, ve bu kural yalnızca bf16 referansa uygulanıyor — kuantize davranış görülmeden.
0.5B, "bu ölçekte kalibrasyon ölçülemez" iddiasının belgesi olarak **taban kontrolü** kalıyor.

### Pilot ifşası
Fizibilite sırasında N=1 bir gözlem yapıldı: 4-bit model yanlış cevaba bf16'dan **daha güvenli**
(0.767 vs 0.687), doğru cevaba daha güvensiz (0.102 vs 0.197). Bu, hipotezlerin yönünü
biliyorduk demektir. **Ön-kaydın §0'ında açıkça yazıldı.** Gizlemek ön-kaydı zayıflatırdı.

---

## Bu projede yakalanan hatalar

### 1. Uydurulmuş kuantizasyon modları "OK" verdi
Keşifte 7 mod denendi, hepsi geçti — ama `banana`, `xyz123` ve boş string de geçti.
**Mod doğrulanmıyordu.** Kaynağı: `nn.quantize` çıplak bir `Linear`'a çağrılınca alt-modül
bulamayıp **sessizce hiçbir şey yapmıyor.**
**Nasıl yakalandı:** Mutasyon testi (PROTOKOL Kural 4). Saçma mod eklemeseydik, ön-kayda
"7 mod destekleniyor" yazacaktık.

### 2. No-op kuantizasyon — parmak izi testi de boştu
İlk parmak izi testinde tüm modlar aynı `sum=0.61` verdi ve tüm attribute'lar `None`'dı.
Sebep aynı: kök modül dönüştürülmüyor.
**Düzeltme:** Modeli bir `nn.Module` sarmalayıcıya koyduk. Sonra `Linear → QuantizedLinear`
dönüşümü, `scales`/`biases` anahtarları ve bit başına farklı parmak izleri göründü —
ve `banana` **reddedildi.**

### 3. `mx.utils` yanlış import
Küçük hata ama testi tamamen sessizleştirdi (`AttributeError` her satırda). Doğrusu
`mlx.utils.tree_flatten`.

### 4. HF cache eksik snapshot
`convert` içeride `local_files_only=True` çağırıyor; `.gitattributes` ve `README.md` eksikse
`IncompleteSnapshotError` atıyor — ama **kuantizasyon zaten tamamlanmış** oluyor, hata
kopyalama aşamasında.
**Kalıcı çözüm:** her `convert` öncesi `snapshot_download`. SPEC ve PREREG'e yazıldı.

### 5. Isınma kirlenmesi (warmup contamination)
3B ilk ölçümde 693 ms/item, ikinci ölçümde **193 ms/item** — 3.6× fark.
Termal throttling değil (0.5B tekrar ölçümde 50ms, stabil). Sebep: MLX ağırlıkları tembel
yüklüyor ve Metal kernel'ları ilk kullanımda derliyor.
**Kalıcı çözüm:** her hücrede ilk 20 item warmup, **atılır.** PREREG §4.4'e yazıldı.

---

## Görev 1 — implementasyon sırasında verilen kararlar (2026-07-25)

### `config.PROMPT_SHA256` eklendi, SPEC §2'nin listelemediği bir sabit
`tests/test_prompt_frozen.py` (YAPILACAKLAR Görev 1) `config.PROMPT_SHA256` ile karşılaştırma
yapıyor, ama SPEC §2'deki `config.py` bloğu bu sabiti içermiyordu — görev tanımı ile spec
arasında bir boşluk. Karar: sabiti ekle (bilim parametresi değil, dondurulmuş şablonun hash'i;
PREREG'e dokunmuyor), SPEC §9 Changelog'a işlendi. `PREREG.md` **düzenlenmedi.**

### `scratch/` .gitignore'da ama keşif script'leri yine de commit'te
SPEC §1: keşif script'leri "repoda kalır, provenance" ve "yalnızca commit'liyse scratch/'e
taşınır." YAPILACAKLAR Görev 1: `.gitignore`'a `scratch/` ekle. İkisi çelişiyor gibi görünüyor
ama değil: `git mv` zaten **izlenen** dosyaları taşıdığında gitignore kuralı onları
izlenmekten çıkarmıyor (gitignore yalnızca *yeni/untracked* dosyalara uygulanır). 14 keşif
script'i `scratch/`'e taşındı ve **hâlâ commit'te** — `git status` ile doğrulandı. `scratch/`
gelecekteki atılabilir dosyalar için gitignore'lu kalıyor.

### `tests/test_determinism.py` pytest formatında değil (AÇIK BULGU olarak kaydedildi)
Görev 1 kapsamında **değiştirilmedi** (SPEC "already passing" diyor, taşıma dışında dokunma
talimatı yok) ama YAPILACAKLAR § AÇIK BULGULAR'a eklendi: assert yok, her `make test` bir
model conversion'ı tetikliyor, pytest 0 test topluyor. Bir sonraki oturumda ele alınmalı.

---

## Görev 2 — metrik testleri, implementasyon YOK (2026-07-25)

### `src/metrics.py` yazılmadı — YAPILACAKLAR'ın açık talimatına uyuldu
Görev 2 metni iki yerde çelişiyordu: "Yap" bölümü `src/metrics.py`'yi **YAZMA** diyor, kabul
kriteri ise "hepsi `NotImplementedError` ile FAIL etmeli" diyor. Modül hiç yoksa bu mümkün
değil — import `ModuleNotFoundError` ile **collection hatası** verir, tek tek test FAIL'i
değil. Çalıştırılıp doğrulandı (`pytest tests/test_metrics.py -q` → "1 error during
collection"). SPEC §7 madde 2 metrics.py'nin bu aşamada var olmadığını teyit ediyor, yani
YAZMA talimatı doğru kabul edildi, stub bile yazılmadı. Karar: kullanıcıya gerçek çıktı
gösterilecek, kabul kriteri metni muhtemelen yanlış yazılmış — düzeltme kullanıcıya bırakıldı.
AÇIK BULGULAR'a eklendi. **Kararın ne zaman verildiği:** sonuçlar (collection hatası)
görülmeden önce, yalnızca YAPILACAKLAR + SPEC metni okunarak.

### Sentetik test verileri numpy-only referans implementasyonlarla doğrulandı
`statsmodels`/`scipy` kurulu değil (requirements.txt'te yok, ayrı bir açık bulgu olarak
kaydedildi). `cal_slope`/`cal_intercept` testlerindeki toleransları (ör. mükemmel kalibrasyonda
1.0±0.05) rastgele seçmek yerine, sıfırdan bir Newton-Raphson lojistik regresyon (numpy-only)
yazıp gerçek sayıları hesapladım ve script'i **committe etmedim** (scratch, `/private/tmp`
altında). Sonuçlar: mükemmel kalibrasyon slope≈1.012/intercept≈0.001, aşırı-uçlu (2× logit)
slope≈0.501, sistematik aşırı-güven (logit+1 kayma) intercept≈-1.012. ECE için de aynı şekilde
`np.array_split` tabanlı referans fonksiyon yazıp N=97'de bin boyutlarının (7,6) en fazla 1
farklı olduğunu, N=50000 mükemmel kalibrasyonda ECE<0.01 olduğunu, dengeli+sabit-0.9 güvende
ECE'nin **tam olarak** 0.4'e eşitlendiğini (her bin'in doğruluğu 0.9'un altında kaldığı için
mutlak değer toplamının teleskopik olarak sadeleşmesiyle) ve dar-aralık/eşit-genişlik-kör
kurgusunda equal-width tek-bin ECE'sinin tam 0.0, equal-mass'ın ise >0.05 çıktığını doğruladım.
**Neden önemli:** kardeş çalışmada tam bu adım atlanmış ve ECE testi sıralı `y` üretildiği için
doğru kodda bile patlamıştı (PROTOKOL Kural 3) — burada tersine, testi yazmadan önce verinin
gerçekten iddia edilen davranışı ürettiğini kanıtladım.

### `bootstrap_ci` dönüş tipi `(lo, point, hi)` tuple olarak varsayıldı
SPEC §3 dönüş tipini belirtmiyor. Test dosyası bu sözleşmeyi `(lo, point, hi)` tuple olarak
sabitliyor — Görev 3 bunu değiştiremez (testi zayıflatmadan geçirme kuralı). Karar sonuçlar
görülmeden, yalnızca yaygın kullanım deseni baz alınarak verildi.

### Kabul kriteri ↔ "YAZMA" çelişkisi çözüldü: imza iskeleti (kullanıcıyla birlikte, 2026-07-25)
Yukarıdaki çelişki kullanıcıya sunuldu ("öneriniz ne?" soruldu). Önerilen ve uygulanan çözüm:
`src/metrics.py`'ye SPEC §3'ün **tam imzalarıyla** bir iskelet yazıldı, her fonksiyon gövdesi
yalnızca `raise NotImplementedError` — sıfır hesap, sıfır mantık. Gerekçe: modülün hiç
var olmaması `ModuleNotFoundError` ile tek bir collection hatası üretir ve hangi testin hangi
çağrı imzasıyla (argüman sırası/adı) uyuşmadığı görünmez; imza hatası ancak Görev 3'te fark
edilirdi. İmza iskeleti hem "implementasyon yazma" yasağını korur (mantık yok) hem de her
testin **kendi FAIL nedeniyle** çalışmasını sağlar. Doğrulandı: `pytest tests/test_metrics.py -q`
→ 13/13 `NotImplementedError`. Bu desen **PROTOKOL Kural 3'e kalıcı ek olarak yazıldı** —
Görev 4/5/6'daki benzer "test önce, implementasyon yok" adımlarında da uygulanacak.
**Kararın ne zaman verildiği:** sonuçlar (13 testin FAIL çıktısı) görülmeden önce.

---

## Görev 3 — `src/metrics.py` implementasyonu (2026-07-25)

### Bağımlılık eklendi: `statsmodels==0.14.6` + `scipy==1.18.0`
Açık bulgu (Görev 2'de kaydedilmişti) çözüldü: `requirements.txt` + `requirements.lock.txt`
aynı commit'te güncellendi (SPEC §0 madde 8). Kural 6 gereği yalnızca `import` ile
yetinilmedi — `sm.GLM(..., offset=..., family=Binomial()).fit()` gerçekten çağrılıp bir
katsayı üretti, doğrulandı (bkz. PROTOKOL Kural 6'daki scipy 1.14 `_lazywhere` hatası
örneği; bu ortamda scipy 1.18 ile sorun yok).

### `tests/test_metrics.py` içinde gerçek bir hata bulundu ve düzeltildi (kullanıcı onayıyla)
`test_cal_slope_perfect_calibration_is_near_one`, referans Newton-Raphson fonksiyonunun
`(b0, b1) = (intercept, slope)` döndürdüğü tuple'ı `expected_slope, _ = ...` şeklinde yanlış
sırayla açıyordu — `expected_slope` değişkenine aslında intercept (≈0.0012) atanıyordu, slope
(≈1.0124) değil. Sonuç: testin **kendi sanity-check satırı** (`assert abs(expected_slope - 1.0)
< 0.05`) implementasyondan bağımsız olarak patlıyordu.

**Nasıl ayırt edildi (implementasyon hatası mı, test hatası mı):** `metrics.cal_slope(y, conf)`
doğrudan çalıştırıldı → `1.0124437146123444`, referans fonksiyonun kendi `b1`'i ile
(`1.012443714612344`) ondalık düzeyinde birebir eşleşti. Yani `src/metrics.py` doğruydu;
hata yalnızca testin tuple unpacking sırasındaydı.

**Karar süreci:** Kendi başıma düzeltmedim — PROTOKOL Kural 3/5 gereği ("test yanlışsa DUR ve
söyle") kullanıcıya kanıtla (iki sayının eşleşmesi) birlikte sunuldu, kullanıcı tek satırlık
unpacking düzeltmesini onayladı. Uygulanan değişiklik: `expected_slope, _ = ...` →
`_, expected_slope = ...` (satır 96). Hiçbir tolerans/eşik gevşetilmedi, hiçbir assertion
silinmedi — yalnızca bir değişken atama hatası düzeltildi. **Kararın ne zaman verildiği:**
sonuç (14/14 yeşil) görülmeden önce, yalnızca iki sayının eşleştiği kanıtlanarak.

**Doğrulama:** `python -m pytest tests/ -q` → 14 passed (13 metrik testi + 1 prompt-freeze
testi; `test_determinism.py` pytest formatında olmadığı için 0 test topluyor, ayrı açık bulgu).
`git diff HEAD -- tests/` artık **boş değil** — Görev 3'ün orijinal kabul kriteri metni ("git
diff BOŞ") bu tek satırlık test-hata-düzeltmesini öngörmüyordu; asıl niyet ("hiçbir test
implementasyona uydurmak için gevşetilmedi") korundu.

---

## Açık bulgu temizliği + protokol geliştirmesi (2026-07-25)

### `tests/test_determinism.py` gerçek pytest testine çevrildi
Görev 1'de kaydedilen açık bulgu: dosya bir top-level script'ti (assert yok, import anında
gerçek bir model conversion'ı çalıştırıyordu, pytest 0 test topluyordu). Sözleşmenin kendisi
(iki koşu bit-bit aynı + 1e-6 gürültü yakalanıyor) daha önce çalıştırılarak doğrulanmıştı;
değişen yalnızca ifade biçimi — `pytest.fixture(scope="module")` ile model bir kez kuruluyor,
iki ayrı test fonksiyonu (`test_repeated_inference_is_bit_identical`,
`test_injected_noise_is_detected`) kendi `assert`'iyle geçiyor/kalıyor. `pytest tests/ -v` →
16 passed (13 metrik + 2 determinizm/mutasyon + 1 prompt-freeze).

### `CLAUDE.md` Adım 2'ye AÇIK BULGULAR görev-numarası taraması eklendi
Önceki oturumda kullanıcıya sorulan bir protokol boşluğu: Adım 2 yalnızca bir görevin kendi
"Ön koşul" satırını (başka bir numaralı görevi referans alan) kontrol ediyordu, ama
`AÇIK BULGULAR`'daki "Engellediği görev: N" etiketli satırları göreve başlamadan **otomatik
taramıyordu** — yani bir bulgu, ilgili görev numarasına doğru şekilde etiketlense bile,
görev başlarken fark edilmeden atlanabilirdi. Kullanıcı onayıyla eklendi: artık Adım 2, o
görev numarasına etiketli işaretlenmemiş bir bulgu varsa, onu da "sağlanmamış ön koşul" gibi
ele alıyor. **Karar ne zaman verildi:** önerildiği oturumda değil, kullanıcının bu oturumdaki
genel "en iyisini yap" onayıyla, sonuç görülmeden önce.

---

## Görev 4 öncesi açık bulgu çözümü (2026-07-25)

### MMLU/ARC-Challenge seçenek sayısı doğrulandı — sabit varsayılamaz
Görev 1'de kaydedilen açık bulgu ("MMLU'da seçenek sayısı her zaman 4 mü, doğrulanmadı"),
Görev 4'ü (`src/benchmarks.py`) engelliyordu. CLAUDE.md Adım 2'deki görev-numarası taramasıyla
yakalandı ve implementasyona başlamadan **çalıştırılarak** çözüldü:

- `cais/mmlu` `all` split, `test` (14042 satır) — hepsinin `choices` uzunluğu taranarak
  tek bir küme elde edildi: `{4}`. MMLU her zaman 4 seçenekli.
- `allenai/ai2_arc` `ARC-Challenge` `test` (1172 satır) — aynı tarama: `{3, 4, 5}`. ARC-Challenge
  **sabit değil.**

**Sonuç:** `src/benchmarks.py::Item.options` zaten bir `list[str]` olarak tasarlanmıştı
(SPEC §3) ve `results/` şemasında ayrı bir `n_options` kolonu var (SPEC §4) — bu tasarım
kararı şimdi veriyle doğrulandı, kod tarafında hiçbir yerde "4 seçenek" sabitlenmeyecek.
`load_items` seçenek sayısını kaynaktan olduğu gibi alacak, `answer_idx`'i o item'ın kendi
`options`/`labels` uzunluğuna göre geçerli aralıkta üretecek.

---

## Görev 5 öncesi açık bulgu çözümü (2026-07-25)

### `mixed_*` recipe çağrı biçimi doğrulandı — `quant_predicate` string, `q_group_size=None`
Görev 1'de kaydedilen açık bulgu ("`mixed_*` recipe'lerin nasıl çağrıldığı doğrulanmadı ·
Engellediği görev: 5"), CLAUDE.md Adım 2'deki görev-numarası taramasıyla yakalandı ve
implementasyona başlamadan **çalıştırılarak** çözüldü. `mlx_lm/convert.py` kaynağı okundu ve
gerçek bir `convert()` çağrısıyla (`Qwen2.5-0.5B-Instruct-bf16`, cache'te hazır) doğrulandı:

- `mlx_lm.convert.QUANT_RECIPES == ["mixed_2_6", "mixed_3_4", "mixed_3_6", "mixed_4_6"]` —
  `config.CONDITIONS`'taki `mixed_*` etiketleriyle **birebir aynı string**. `condition_tag`
  doğrudan `quant_predicate=` argümanına geçirilebiliyor, ayrı bir eşleme tablosu gerekmiyor.
- `quant_predicate` bir `str` olduğunda `convert()` içeride `q_mode != "affine"` ise
  `ValueError` fırlatıyor — yani recipe çağrılarında `q_mode="affine"` **zorunlu**
  (mxfp4/mxfp8 recipe yok, PREREG §4.1'deki "mixed recipes — mode: —" satırıyla tutarlı).
- `q_group_size` recipe çağrısında `None` geçilmeli (PREREG §4.1 tablosundaki "group_size:
  default" tam olarak bunu ifade ediyor). `None`, `mixed_quant_predicate_builder`'ın kendi
  varsayılanını (`64`) **ezmiyor** çünkü zincirin sonunda `mx.core.quantize(group_size=None,
  ...)` `affine` modu için kendi iç varsayılanı 64'ü uyguluyor — `config.json`'da
  `quantization.group_size == 64` olarak doğrulandı (üst seviye), per-layer sözlüklerde
  `group_size: None` görünmesi kozmetik (predicate'in döndürdüğü ham değer), gerçek
  kuantizasyona etkisi yok.
- `q_bits=None` de aynı şekilde geçilmeli — `quant_predicate` her katman için kendi
  `{"bits": ...}` sözlüğünü döndürdüğünden üst seviye `bits` argümanı per-layer override'ları
  etkilemiyor.
- Uçtan uca doğrulama: `mixed_2_6` ile dönüştürülen 0.5B modelin `config.json`'ında 169
  per-layer girişte **yalnızca `{2, 6}`** bit değerleri görüldü (recipe adıyla birebir
  eşleşiyor), model `mlx_lm.load()` + `generate()` ile **gerçekten çalıştırıldı** (çökme yok).
  Konsol logu `"[INFO] Quantized model with 2.937 bits per weight."` satırını üretti — SPEC'in
  istediği "efektif bit dönüşüm logundan parse edilir" kuralı bu recipe için de geçerli ve
  test edildi.

**Sonuç:** `src/quantize.py::build`'ın `recipe` dalı, `condition_tag`'i doğrudan
`quant_predicate=` olarak geçiriyor, `q_mode="affine"`, `q_group_size=None`, `q_bits=None`.

### Warmup semantiği: 20 item **ek**, 1000'den **düşülmüyor**
SPEC §3 `measure.run_cell` metni ("ilk `N_WARMUP` item çalıştırılır ve atılır... sayılmaz")
tek başına iki okumaya açıktı: (a) toplam 1000 item'ın ilk 20'si warmup olur, geriye kalan
980'i skorlanır; (b) warmup 1000'in **dışında**, aynı ilk-20 item iki kez çalıştırılır (önce
ısınma, sonra gerçek skor), skorlanan toplam yine 1000 kalır. SPEC §4'teki meta şeması
(`"n_items_scored": 1000, "n_warmup_discarded": 20` — **ayrı ayrı** alanlar, biri diğerinden
düşülmüyor) okuması (b)'yi doğruluyor: `N_ITEMS=1000` her hücrede **skorlanan** sayıdır,
warmup ekstra bir maliyettir. **Karar, sonuç görülmeden önce**, yalnızca SPEC §4 şema
tanımına dayanarak verildi: `run_cell`, `items[:N_WARMUP]`'ı önce (atılarak) çalıştırıyor,
sonra **tüm** `items` listesini (1000'in tamamı, ilk 20 dahil) skorluyor. Görev 6'da
`tests/test_no_leakage.py` bu sözleşmeyi (`n_scored == n_items`) mekanik olarak kilitleyecek.

### Fonksiyonel doğrulama (Kural 6) — üç mod da gerçek dönüşümle test edildi
`src/quantize.py::build` her üç dal için (bf16, affine, mxfp4) ve `recipe` dalı için cache'teki
`Qwen2.5-0.5B-Instruct-bf16` ile gerçekten çalıştırıldı: `affine_b4_g64` → 4.501 bit (Görev 5
kabul kriteri, `1.5b` ile), `bf16` → cache yolu döndü (dönüştürme yok, `effective_bits=16.0`),
`mxfp4_b4_g32` → 4.252 bit, `mixed_2_6` → 2.937 bit. `src/measure.py::run_cell` da aynı
dönüştürülmüş 0.5B model + 30 ARC-Challenge item'ıyla çalıştırıldı: 30 satır, 0 `status="failed"`,
doğruluk ~%37 (0.5B'nin beklenen düşük-doğruluk aralığında, GECMIS'teki "0.5B ana ızgaradan
çıkarıldı" notuyla tutarlı).

---

## Kardeş ML çalışmasından devralınan dersler

`~/github-projects/imbalance-calibration` — tamamlandı, yayında. Orada yakalanan yedi hata,
PROTOKOL.md'nin 11 kuralını doğurdu. Bu projede baştan uygulananlar:

| ML'de yaşandı | Burada baştan engellendi |
|---|---|
| Ön-kayıt, veri kaynağına sorgu atılmadan yazıldı → 2 sapma | **Fizibilite kapısı önce** (Kural 2) |
| İki sızıntı testi totolojikti | **Mutasyon zorunluluğu** (Kural 4) |
| `bootstrap_ci` kwargs iletmiyordu | SPEC'e açıkça yazıldı, test edilecek |
| ECE testi sıralı `y` ile bozuktu | YAPILACAKLAR Görev 2'de uyarı olarak duruyor |
| `import X` başarılı, `import X.api` çöküyordu | **`make verify` fonksiyonel olacak** (Kural 6) |
| Ajan "yaptım" dedi, başka klasöre yazıyordu | **Her görevde kanıt komutu** (Kural 5) |
| Repo adı bozuktu (slash/parantez) → hayalet klasör | Temiz isim baştan |
| Cursor co-author trailer'ı → 2 repo temizlendi | Her commit sonrası kontrol |

---

## Doğrulanmış teknik olgular (hepsi çalıştırılarak)

- `affine` bits: **2, 3, 4, 5, 6, 8** — her biri farklı parmak izi
- `affine` group_size: **32, 64, 128**
- `mxfp4` / `mxfp8`: **group_size 32 zorunlu**, diğerleri açık hatayla reddediliyor
- `nf4`: **YOK** (`KeyError`)
- `dynamic_quant`: **YOK** (imzada ve `QUANT_RECIPES`'te yok)
- `QUANT_RECIPES`: `mixed_2_6`, `mixed_3_4`, `mixed_3_6`, `mixed_4_6`
- Nominal 4-bit = **4.501–4.502 efektif bit/ağırlık** (scales + biases yüzünden)
- Inference **deterministik** — iki koşu bit-bit aynı, 1e-6 gürültü bile yakalanıyor
- Hız (steady state): 0.5B 50ms · 1.5B 127ms · 3B 193ms
- Doğruluk (30 item ARC-C): 0.5B %33 · 1.5B %73 · 3B %73
- Termal throttling **yok**
- Benchmark boyutları: ARC-Easy 2376 · ARC-Challenge 1172 · MMLU 14042
- HellaSwag/PIQA: farklı protokol (devam skorlama) → elendi
- PIQA ayrıca `datasets 5.0.0`'da loading-script desteği kalktığı için yüklenmiyor

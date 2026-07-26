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

## Görev 6 — `tests/test_no_leakage.py` + mutasyon kanıtı (2026-07-25)

### Açık bulgu: `runner.py` yokken sızıntı/sıra testleri nasıl yazılır
SPEC §7 build order, `test_no_leakage.py`'yi (madde 6) `runner.py`'den (madde 7) **önce**
sıralıyor, ama testin kendi tanımı ("Faz 3, eligibility.json yokken çalışamaz", "uygunluk
yalnızca bf16'dan") `runner.py`'de bir şeye karşı test edilmeyi gerektiriyor — modül hiç
yoksa test totolojik ya da imkansız olurdu. **AÇIK BULGULAR'a eklendi** ("Görev 6:
runner.py henüz yok, sızıntı/sıra testleri neye karşı yazılacak? · Engellediği görev: 6 ·
Ciddiyet: orta"), sonra **aynı oturumda çözüldü**: `src/runner.py`'ye SPEC §3'e eklenen iki
yeni saf fonksiyonla (`compute_eligibility`, `assert_phase3_allowed`) **kısmi** bir
implementasyon yazıldı — `cell_id` ve `run_all` (gerçek `NotImplementedError` iskeleti)
Görev 7'de kalıyor. Gerekçe: bu iki fonksiyon hiçbir kuantizasyon parametresi seçmiyor/
değiştirmiyor (SPEC §0 madde 1 riski yok), yalnızca kapı/muhasebe mantığı — Kural 3 ek
deseninin ("imza iskeleti") aksine burada iskelet değil **gerçek implementasyon** yazmak
daha doğru seçim, çünkü asıl istenen (mutasyonla kanıtlanmış sızıntı koruması) ancak çalışan
bir implementasyona karşı mümkün. SPEC §9 Changelog'a işlendi. **Kararın ne zaman verildiği:**
sonuçlar (testlerin geçmesi) görülmeden önce, yalnızca SPEC §7 sırası + SPEC §0 madde 1
okunarak.

### `compute_eligibility`'nin girdi tasarımı: tam `cells` DataFrame'i, önceden filtrelenmiş dict değil
İlk tasarım seçeneği, fonksiyonun zaten bf16-only bir `{model: {benchmark: accuracy}}`
sözlüğü almasıydı — ama bu, "sızıntı" riskini test edilemez hale getirirdi: girdi zaten
filtrelenmişse, fonksiyonun kendisi hiçbir zaman yanlış filtreleyemez. PREREG §4.2'nin asıl
endişesi ("uygunluk kuantize davranış görülmeden önce karar verilmeli") ancak fonksiyon
**tüm hücrelerin** (bf16 + kuantize karışık) ham sonuçlarını alıp kendi içinde
`condition == "bf16"` filtresi uyguladığında anlamlı şekilde test edilebilir. Bu yüzden
imza `compute_eligibility(cells: pd.DataFrame)` olarak değiştirildi — SPEC §3'e bu haliyle
yazıldı.

### Mutasyon kanıtı — üç ayrı sözleşme, üçü de doğru sebeple patladı (PROTOKOL Kural 4)
Üç geçici bozuk implementasyon yazıldı, ilgili teste karşı koşuldu, çıktı gösterildi, sonra
**satır satır orijinaline geri döndürüldü** (`git diff HEAD -- src/measure.py` boş; `runner.py`
henüz commit'te olmadığı için `git diff` izlemiyor, ama düzeltme öncesi/sonrası dosya içeriği
elle karşılaştırılarak doğrulandı).

1. **Warmup sızıntısı** (`src/measure.py::run_cell`): warmup döngüsü, skorlanan satırları da
   `rows`'a ekleyecek şekilde bozuldu. `test_scored_row_count_equals_input_items_not_more` →
   `45 == 25` ile FAIL (25 item yerine 45 satır — ilk 20 item iki kez göründü);
   `test_no_duplicate_item_ids_in_output` → `t0`..`t19`'un iki kez göründüğü gösterilerek FAIL.
2. **Uygunluk sızıntısı** (`src/runner.py::compute_eligibility`): `bf16 = cells[cells["condition"]
   == "bf16"]` satırı `bf16 = cells` ile değiştirildi (filtre kaldırıldı).
   `test_eligibility_ignores_quantized_rows_for_the_same_model` → beklenen `0.3` yerine
   `0.6316` (bf16 + kuantize karışımı) ile FAIL — yani kirlenmiş doğruluk kanıtlandı.
3. **Sıra kapısı devre dışı** (`src/runner.py::assert_phase3_allowed`): `if not path.exists()`
   → `if False` yapıldı (kapı hep açık). `test_phase3_blocked_without_eligibility_file` →
   `DID NOT RAISE RuntimeError` ile FAIL.

Üçü de **kendi doğru nedeniyle** patladı (yanlış assertion, yanlış tolerans değil). Bozuk
kodlar silinip orijinal haline getirildi; `pytest tests/ -q` → 33 passed (25 önceki + 8 yeni,
regresyon yok).

---

## Görev 7 — `src/runner.py::run_all` (2026-07-25)

### `run_all` tasarımı
Üç fazlı sıra `_run_and_write_cell` adlı tek bir yardımcı fonksiyonla uygulandı (build ->
measure -> parquet+meta yaz -> teardown, `try/except Exception` ile sarılı — asla `raise`
etmiyor, SPEC §0 madde 2). Faz 2, Faz 1'in bellekteki sonuçlarını değil, **diskteki**
`results/cells/*.parquet` dosyalarını okuyarak `compute_eligibility`'ye veriyor — böylece
yarıda kesilip devam ettirilen bir koşu, önceki oturumdan kalan bf16 hücreleriyle de doğru
uygunluk hesaplar (SPEC §5 resumability). Faz 3, `eligible=True` **veya** `role=
"floor_control"` olan modelleri alıyor (PREREG §4.2: taban kontrolü kendi uygunluk kararından
bağımsız olarak tam merdiveni koşar).

`pd.DataFrame({"is_correct":...}).mean()` üzerinde `status="failed"` satırlarının
`is_correct=None` değerlerinin `compute_eligibility`'yi bozup bozmadığı **çalıştırılarak**
doğrulandı (Kural 6): pandas object-dtype `.mean()` `None`'ları NaN gibi atlıyor, hata
fırlatmıyor — yani bir item'ın ağ/model hatasıyla başarısız olması, o hücrenin bf16 doğruluk
hesabını bozmuyor, sadece payda küçülüyor.

### Açık bulgu: `make pilot` bir CLI/model beklerken hiçbiri tanımlı değildi
`Makefile`'daki `pilot:` hedefi (Görev 1'de yazılmış) `$(PYTHON) -m src.runner --pilot`
çağırıyor, ama SPEC §3'ün `runner.py` sözleşmesi yalnızca `run_all(force)`'ı tanımlıyor —
`__main__` girişi yok, `--pilot` bayrağı yok. Ayrıca SPEC §8/YAPILACAKLAR Görev 8 "bir model"
diyor ama modelin adı hiçbir belgede geçmiyor. **AÇIK BULGULAR'a eklendi** ("Engellediği görev:
8"), aynı oturumda çözüldü: `run_pilot()` eklendi — SPEC §8'deki 3 hücreyi (bf16,
affine_b4_g64, affine_b2_g64 × arc_challenge) `run_all` ile **aynı** üç-fazlı sırayla
(`_run_and_write_cell`/`compute_eligibility`/`assert_phase3_allowed` yeniden kullanılarak)
koşturuyor; `run_all`'ın SPEC'teki imzası değişmedi. Model seçimi `config.PILOT_MODEL =
"qwen2.5-1.5b"` olarak `config.py`'ye eklendi (SPEC §9 Changelog) — dört ana modelin en hızlısı
ve keşif fazında %73 doğrulukla uygunluk barının belirgin üzerinde; bilinçli mühendislik
kararı, PREREG'e dokunmuyor (pilot verisi §0 Pilot İfşası gereği zaten nihai tablolara
girmiyor). **Kararın ne zaman verildiği:** sonuçlar görülmeden önce, yalnızca hız/uygunluk
marjı gerekçesiyle.

Pilot hücrelerinin normal `cell_id` altında normal `results/` ağacına yazılması **bilinçli**:
tam koşunun Faz 1 cache'i (SPEC §5) bu hücreleri yeniden hesaplamak yerine kullanır; tam
koşunun kendi Faz 2'si tüm bf16 hücrelerinden `eligibility.json`'ı zaten yeniden üretir, yani
pilotun tek-model'lik `eligibility.json`'ı kalıcı değil, geçici bir yan üründür.

### Fonksiyonel doğrulama (Kural 6 — varlık değil, çalıştırma)
`inspect.getsource` tabanlı kabul kriteri (YAPILACAKLAR'daki komut) yalnızca statik bir
kontrol. Ek olarak `runner._run_and_write_cell` cache'teki `qwen2.5-0.5b` (floor control,
dönüştürme gerektirmiyor) ile **gerçekten** çağrıldı (geçici `scratchpad/runner_smoke/`
dizinine, `results/`'a dokunulmadan): 1000 satırlık doğru şemalı parquet yazıldı
(`n_failed_items=0`), meta json SPEC §4 alanlarıyla eşleşti, ikinci çağrı cache nedeniyle
atlandı (meta dosyasının `mtime`'ı değişmedi), `compute_eligibility` gerçek veriden
`{"qwen2.5-0.5b": {"arc_challenge": 0.518, "eligible": True, "role": "floor_control"}}`
üretti, `assert_phase3_allowed` `raise` etmedi. Not: bu 0.518, feasibility fazındaki 30 item'lık
%33 tahmininden belirgin farklı (1000 item'lık gerçek örneklem) — bir kod hatası değil, örneklem
büyüklüğü farkı; floor control modelin tam merdiveni her koşulda çalıştığı için (yukarıya bkz.)
bu sayı Faz 3'ün davranışını etkilemiyor. Test dizini sonra silindi.

---

## Görev 8 — `make pilot` çalıştırıldı, mkdtemp/convert açık bulgusu (2026-07-25)

### İlk koşu: her iki kuantize hücre `status="failed"`
`make pilot` ilk çalıştırıldığında bf16 `status="ok"` (206.7s) verdi ama `affine_b4_g64` ve
`affine_b2_g64` ikisi de anında (`wall_seconds` < 0.4s) `status="failed"` döndü, `n_items_scored: 0`.
Kök neden: `runner._run_and_write_cell`, `out_dir`'i `tempfile.mkdtemp(prefix=...)` ile önceden
**oluşturuyordu**, ama `mlx_lm.convert()` kaynağı hedef dizin zaten varsa `ValueError` fırlatıyor
("Cannot save to the path ... as it already exists"). Görev 5'in fonksiyonel doğrulaması bunu
yakalamamıştı çünkü orada `quantize.build()` doğrudan, `mkdtemp` kullanmayan sabit bir yolla
(`/tmp/q_t1`, önceden var olmayan) çağrılmıştı — hata yalnızca `runner.py`'nin orkestrasyon
yolundan geçince ortaya çıktı. AÇIK BULGULAR'a **kritik** olarak eklendi.

### Çözüm
`mkdtemp` ile tekil bir isim üretilip hemen `os.rmdir` ile siliniyor — `convert()`'e her zaman
**yeni ama garantili tekil** bir yol veriliyor (küçük bir TOCTOU penceresi var, ama tek süreçli
araştırma kodu için kabul edilebilir). `make pilot` yeniden koşuldu: 3/3 hücre `status="ok"`.

### Sağlık kontrolü (kullanıcıya gösterildi, onaylandı)
| condition | acc | mean_conf | ECE | Brier |
|---|---|---|---|---|
| bf16 | 0.765 | 0.896 | 0.130 | 0.168 |
| affine_b4_g64 | 0.737 | 0.873 | 0.136 | 0.177 |
| affine_b2_g64 | 0.244 | 0.742 | 0.498 | 0.459 |

2-bit hücresi neredeyse şans seviyesine çöktü ve ECE'si bf16'nın ~4 katı — kuantizasyon
gerçekten uygulanıyor, `nn.quantize`'ın çıplak `Linear`'da sessiz no-op olma riski (keşifte
yaşanmıştı) burada gerçekleşmedi.

### Süre bütçesi
3 hücrenin ortalaması (en hızlı model, tek benchmark) ~199s/hücre → 140 hücreye ölçeklenirse
~7.75 saat, 8 saatlik sınıra çok yakın. Kullanıcıya bildirildi, tam koşuya onay alındı.

---

## Görev 9 — tam koşu, kesinti/devam testi, Llama BOS token hatası (2026-07-25 → 26)

### Kesinti/devam testi (SPEC'in kendi kabul kriteri)
`caffeinate -i make reproduce` başlatıldı, ~5 dk sonra (`test` bitmiş, `runner` bir bf16 hücresi
ortasındayken) kasten `TaskStop` ile kesildi — henüz hiçbir yeni hücre diske yazılmamıştı.
Yeniden başlatıldığında pilotun 3 hücresinin parquet `mtime`'ları **değişmedi** — tamamlanmış
hücreler gerçekten atlanıyor, yeniden hesaplanmıyor. Test geçti.

### İlk koşu sonucu: 88/88 `status="ok"`
Grid tamamlandı, `src.analyze` adımında beklenen `ModuleNotFoundError` ile durdu (Görev 10 henüz
yok — veri kaybı yok, runner zaten hücre başına diske yazıyor). 88 hücre, 140 değil: o andaki
`eligibility.json`'da `llama3.2-1b`/`llama3.2-3b` ikisi de `eligible=false` çıkmıştı, bu yüzden
yalnızca bf16'ları (2'şer hücre) çalıştı.

### Açık bulgu: Llama-3.2 tokenizer'ı `encode()`'a otomatik BOS ekliyor
Koşu sürerken `eligibility.json`'da `llama3.2-1b` ve `llama3.2-3b` için **birebir aynı** sayılar
görüldü (`arc_challenge: 0.222`, `mmlu: 0.238`) — iki farklı boyutlu modelin bire bir aynı
doğruluğu vermesi şüpheli bulundu, araştırıldı. `llama3.2-1b__bf16__arc_challenge.parquet`
kontrol edildi: `pred_idx` 1000/1000 satırda sabit `0`, `conf_pred` std=0.0 — model her zaman
"A" seçiyormuş gibi görünüyordu.

Kök neden `transformers.AutoTokenizer` ile doğrudan karşılaştırılarak **kanıtlandı**:
```
LLAMA: encode(" A") == [128000, 362]   # 128000 = <|begin_of_text|>, otomatik ekleniyor
QWEN:  encode(" A") == [362]           # BOS yok
```
`measure._score_item`, `tokenizer.encode(" " + label)[0]` ile (encode çıktısının **ilk**
token'ı) seçenek harfinin logit indeksini alıyordu. Qwen'de doğru (tek token dönüyor), ama
Llama'da dört seçenek için de **aynı sabit BOS token'ının** (128000) logit'i okunuyordu — dördü
eşit çıkıyor, softmax sonrası hepsi 0.25, `argmax` ilk elemanı (index 0) seçiyor. Qwen ailesi
etkilenmedi (tokenizer'ı BOS eklemiyor, sayıları zaten çeşitli/pilot sonuçlarıyla tutarlıydı).
AÇIK BULGULAR'a **kritik** olarak eklendi.

**Karar (sonuçlar görülmeden önce):** arka planda süren koşu, o an yalnızca uygun bulunan Qwen
hücrelerini işlediği için (Llama zaten `eligible=false` olduğundan Faz 3'e hiç girmiyordu) daha
fazla israf yoktu — kullanıcıyla birlikte karar verildi: koşu **durdurulmadı**, düzeltme ayrı bir
adım olarak ele alındı.

### Düzeltme (test-önce + mutasyon kanıtı, PROTOKOL Kural 3+4)
`tests/test_measure.py` yazıldı — gerçek model yüklemeden, iki sahte tokenizer'la
(`_FakeTokenizerWithLeadingBOS`, `_FakeTokenizerWithoutBOS`) `measure._option_token_ids`'i test
ediyor. İlk çalıştırıldığında beklenen `ImportError` verdi (fonksiyon henüz yoktu). Kullanıcıya
gösterildi, onay alındı.

`src/measure.py`'ye saf `_option_token_ids(tokenizer, labels)` yardımcı fonksiyonu çıkarıldı:
`encode(" " + label)[-1]` — etiketin kendi token'ı, başta BOS olsa da olmasa da her zaman **son**
eleman. `_score_item` bu fonksiyonu kullanacak şekilde güncellendi.

**Mutasyon kanıtı:** `[-1]` kasten `[0]`'a geri döndürüldü, BOS testi doğru sebeple FAIL etti
(`assert 128000 not in [128000, 128000, 128000, 128000]` — canlı ortamda görülen artefaktın
birebir aynısı), dosya orijinaline döndürüldü.

**Fonksiyonel doğrulama (Kural 6):** gerçek Llama-3.2-1B tokenizer'ı yüklenip çağrıldı:
`_option_token_ids(tok, ['A','B','C','D']) == [362, 426, 356, 423]` (dördü farklı, önceki
`[128000]*4` yerine). `pytest tests/` → 35 passed.

### Düzeltme koşusu ve nihai sonuç
İlk koşu tamamen bittikten sonra (ikinci bir model-yükleyen süreç artık güvenli) 4 eski/bozuk
llama bf16 dosyası (parquet+meta) ve `eligibility.json` silindi, `python -m src.runner` tekrar
çalıştırıldı: Qwen'in 84 hücresi cache'ten atlandı (dosyalar zaten vardı), yalnızca 4 llama bf16
hücresi düzeltilmiş kodla yeniden ölçüldü, Faz 2 `eligibility.json`'ı doğru sayılarla yeniden
yazdı, Faz 3 yeni uygunluk kararına göre devam etti.

**Nihai sayılar:**
| model | arc_challenge | mmlu | eligible |
|---|---|---|---|
| llama3.2-1b | 0.475 | 0.435 | false |
| llama3.2-3b | 0.713 | 0.569 | **true** |
| qwen2.5-0.5b | 0.518 | 0.436 | true (floor_control) |
| qwen2.5-1.5b | 0.765 | 0.571 | true |
| qwen2.5-3b | 0.81 | 0.647 | true |

`llama3.2-3b` düzeltilmiş ölçümle eşiği geçti — bozuk veride (0.222/0.238) bu model **tamamen
elenmiş** olacaktı, ön-kaydın kendi öngördüğü "havuz 4'ün altına düşebilir" riski (2026-07-24
tarihli açık bulgu) neredeyse gerçekleşiyordu, ama gerçek neden model kalitesi değil ölçüm
hatasıydı. Tam merdiveni işlendi (28 hücre). `llama3.2-1b` hâlâ eşiğin altında (yalnızca bf16,
2 hücre) — bu, gerçek bir sonuç, bir hata değil.

**Nihai grid: 114/114 hücre `status="ok"`, hiç `failed` yok.** Model başına: `qwen2.5-0.5b` 28,
`qwen2.5-1.5b` 28, `qwen2.5-3b` 28, `llama3.2-3b` 28, `llama3.2-1b` 2.

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

## Görev 10 — `src/analyze.py`, PREREG §5 tabloları + figürler (2026-07-26)

### SPEC'e yeni sözleşme eklendi (Değişiklik değil, boşluk doldurma)
`src/analyze.py`'nin SPEC §3'te hiç sözleşmesi yoktu — §1 dosya haritasında adı geçiyordu, §7
madde 10 sadece "dört tablo, üç figür" diyordu, ama H1-H4'ün PREREG §3'teki düzyazı falsifikasyon
kriterlerini ("differencing per item", "intervals overlap", "falls outside that range") tam
olarak hangi algoritmanın uygulayacağı hiçbir yerde yazılı değildi. Bu oturumda SPEC §3'e altı
saf fonksiyon eklendi (`_intervals_overlap`, `_paired_bootstrap_delta`, `_h1_ladder_verdict`,
`_h2_direction_verdict`, `_h3_mode_verdict`, `_h4_recipe_verdict`) — PROTOKOL Kural 3 gereği
**önce test, sonra onay, sonra implementasyon** sırasıyla, iki ayrı onay turunda (H2'nin verdict
fonksiyonu ilk incelemeden sonra eksik olduğu fark edilip ayrıca onaylandı). `tests/test_analyze.py`
27 bilinen-cevap testi içeriyor; hepsi önce imza iskeletine karşı `NotImplementedError` ile
FAIL etti (kullanıcıya gösterildi), sonra implementasyona geçildi, 27/27 geçti.

**Mutasyon kanıtı (Kural 4):** `_intervals_overlap`'in sınır dahil karşılaştırması (`<=`) kasten
sıkı eşitsizliğe (`<`) çevrildi — yalnızca sınırda-değen-aralık testi (`test_intervals_overlap_true_when_touching_at_boundary`)
doğru sebeple FAIL etti (diğer 26 test etkilenmedi), sonra dosya orijinaline döndürüldü.

### Tasarım kararları (PREREG'in düzyazısını algoritmaya çeviren, bilim değiştirmeyen kararlar)
- **Ana tablolar yalnızca uygun modelleri kapsıyor:** `results/eligibility.json`'dan
  `eligible=True` ve `role != "floor_control"` olan modeller (şu an: `qwen2.5-1.5b`,
  `qwen2.5-3b`, `llama3.2-3b`). `llama3.2-1b` (eligible=false) ve `qwen2.5-0.5b` (taban kontrolü)
  Tablo 1-4'e hiç girmiyor — PREREG §4.2 "taban kontrolü ayrı raporlanır" kuralı, kendi dosyası
  `table5_floor_control.csv`'de (dört ana tablodan biri değil, ayrı bir zorunlu ifşa).
- **H1 monotonluk kararı:** bir adımda ECE noktası düşerse (bit azalırken beklenenin tersi),
  bu yalnızca iki komşu koşulun %95 aralıkları **örtüşmüyorsa** gerçek bir ihlal sayılıyor;
  örtüşen aralıklardaki düşüş gürültü kabul ediliyor (H1'in kendi falsifikasyon cümlesi:
  "with 95% intervals excluding the reversal being noise").
- **H2 yön kararı:** her koşul hücresi ("model x benchmark x quantized condition") bf16'ya karşı
  eşleştirilmiş bootstrap delta'sıyla (`_paired_bootstrap_delta`, aynı item indeksleri her iki
  kolda da yeniden örnekleniyor — PREREG §4.5 "differencing per item") değerlendiriliyor;
  genel H2 yalnızca **hiçbir hücre ters yönde anlamlı değilse VE en az bir hücre yönü anlamlı
  şekilde doğruluyorsa** PASS.
- **H3 kararı:** falsifikasyon yalnızca **her** model×benchmark hücresi örtüşüyorsa geçerli;
  tek bir hücrenin bile anlamlı farkı H3'ü ayakta tutuyor (PREREG'in "across all models and
  benchmarks" ifadesi).
- **H4 kararı:** bileşen aralığı ([a,b] bit-genişliğinin ECE **nokta tahminleri**, kendi
  aralıkları değil) ile recipe'nin **kendi** %95 aralığının örtüşüp örtüşmediğine bakılıyor.

### Açık bulgu (kod, bilim değil): `overconfidence_rate` sıfıra bölme
Gerçek 114 hücrelik veride ilk koşuda `qwen2.5-3b`'nin `affine_b2_g64` ve `mixed_2_6`
hücrelerinde (her iki benchmark'ta da bazıları) `numpy` "Mean of empty slice" uyarısı çıktı.
Kök neden **kod hatası değil**: bu hücrelerde `conf_pred` hiçbir item'da `OVERCONF_THRESHOLD`
(0.90)'ı aşmıyor (`max(conf_pred)=0.838`), yani `metrics.overconfidence_rate`'in payda kümesi
boş — metrik o hücrede **tanımsız**, NaN. Bu, aşırı sıkıştırmada modelin hiç %90+ güvenli tahmin
üretmediğini gösteren gerçek bir bulgu (aşırı-güven değil, tam tersi — belirsizlik). `metrics.py`
(Kural 4 kanıtlı, dokunulmadı) değiştirilmedi; bunun yerine `analyze.build_table2`'ye
`overconfidence_rate_n_qualifying` kolonu eklendi (eşiği aşan item sayısı) — NaN artık açıklamasız
görünmüyor, 0 görülünce sebebi tabloda okunuyor. Analiz tekrar koşulunca (aynı koşullar, aynı
sonuç) bu 4 satır dışında hiç NaN yok (`delta_*_lo/hi` sütunlarındaki 6'şar NaN ise bf16
satırlarının kendi kendine deltasının tanımsız olması, kasıtlı — `None` olarak yazıldı).

### Bağımlılık eklendi
`matplotlib==3.11.1` — üç figür için gerekli, önceden yoktu. `requirements.txt` +
`requirements.lock.txt` aynı commit'te güncellendi (SPEC §0 madde 8).

### Fonksiyonel doğrulama (Kural 6) — gerçek 114 hücrelik veriyle iki tam koşu
`python -m src.analyze` gerçek `results/` üzerinde iki kez koşuldu (~9.5 dk her biri, 2000
bootstrap resample × 4 metrik × onlarca hücre). İlk koşu overconfidence_rate bulgusunu ortaya
çıkardı, kod düzeltildi, ikinci koşu temiz çıktı (uyarı yok). Çıktılar elle incelendi:
- Tablo 1-4 + `table5_floor_control.csv` + `verdicts.json` beklenen şekilde yazıldı, NaN'lar
  yalnızca tasarım gereği (bf16'nın kendine deltası) açıklanabilir.
- Üç figür (`figure1_calibration_curves.png`, `figure2_ece_vs_effective_bits.png`,
  `figure3_confidence_distribution.png`) görsel olarak incelendi — Figür 2 beklenen şekli
  gösteriyor (ECE ~4.5 efektif bite kadar düz, altında keskin sıçrama); Figür 1'de 2-bit eğrisi
  diyagonalin belirgin altında (aşırı-güven); Figür 3'te bf16'nın yüksek-güven/doğru
  yoğunlaşması düşük-bit'te dağılıyor. Hiçbiri anormal/hatalı görünmüyor.
- **H1 gerçek veride bir gerçek ihlal yakaladı** (kurgusal test değil): `qwen2.5-3b` hem
  arc_challenge hem mmlu'da 3-bit'te ECE'nin 4-bit ve 2-bit'ten **çok daha kötü** olduğu
  (ör. arc: 3-bit ECE 0.434 [0.404,0.463] vs 2-bit 0.254 [0.226,0.281], aralıklar örtüşmüyor)
  istatistiksel olarak anlamlı bir ters-U görüldü — `_h1_ladder_verdict` bunu doğru şekilde
  ihlal olarak işaretledi, H1 bu iki model×benchmark hücresinde FAIL. Bu bir kod hatası değil,
  raporlanacak gerçek bir bilimsel bulgu (Görev 13'e not).
- `pytest tests/ -q` → **62 passed** (35 önceki + 27 yeni, regresyon yok).

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

---

## Görev 12 — `README.md` (2026-07-26)

`README.md` yazılırken içindeki her manşet sayı `results/tables/*.csv` + `eligibility.json`'dan
birebir okunarak/hesaplanarak dolduruldu (bootstrap CI'lı ECE değerleri `table1_h1_bit_ladder.csv`'den,
2-bit doğruluk yüzdeleri gerçek parquet'ten `is_correct.mean()` ile — ilk taslakta bu üçü tahmindi,
fark edilip gerçek hesapla değiştirildi).

**Yakalanan hata:** İlk taslakta "kardeş çalışma" referansı için uydurma bir `https://github.com`
URL'i yazılmıştı — CLAUDE.md'nin sistem talimatı programlamayla ilgisiz URL uydurmayı yasaklıyor.
Yazarken fark edilip yerel dizin yoluna (`~/github-projects/imbalance-calibration`) düzeltildi,
commit edilmeden önce. Kayıt altına alınıyor çünkü PROTOKOL Kural 5'in ("iddiaya değil koda bak")
bir uzantısı: kendi çıktımı da aynı şüphecilikle kontrol etmem gerekiyor.

---

## Görev 13 — `paper.md` (2026-07-26)

### Related work araştırması gerçekten yapıldı (Kural 1)
6 kaynak `WebSearch` ile bulundu, hepsi `WebFetch` ile gerçek arXiv sayfasından okunarak
doğrulandı (en yakın önceki çalışma için, arXiv:2405.00632, ilk PDF fetch'i başarısız olunca
`arxiv.org/html/...` HTML sürümünden tam metin okundu). Her alıntı kaynağın gerçek metninden
birebir substring seçilerek yazıldı, sonra ayrı bir Python script'iyle kelime sayıları
programatik doğrulandı (hepsi ≤15 kelime; ilk taslakta MLX runtime karşılaştırma makalesinden
alınan bir "alıntı" aslında bir metrik listesiydi ve 19 kelimeydi — tırnak işaretleri kaldırılıp
dolaylı anlatıma çevrildi).

### Yakalanan hata: yazar isimleri doğrulanmadan yazılmıştı
İlk taslakta 6 kaynaktan 4'ünün yazar adını (soyadını arXiv ID'sinden/arama özetinden tahmin
ederek) **yanlış** yazmıştım: "Kwon vd." (gerçeği: Proskurina, Brun, Metzler, Velcin — arXiv
2405.00632), "Bhat & Chen" (gerçeği: Rajesh, Jodhpurkar, Anbuselvan, Singh, Jallepali, Godbole,
Sharma, Shrivastava — arXiv 2511.05502), "Xiong vd." (gerçeği: K. Tian, Mitchell, Zhou, Sharma,
Rafailov, Yao, Finn, Manning — arXiv 2305.14975), "Zhao vd." (gerçeği: Zhou, Zhang, Hu, Li,
Collier, Vulić, Korhonen — arXiv 2510.03136). Referans listesini yazmadan hemen önce her kaynak
için ayrıca "list the full author names exactly as shown on this page" diye `WebFetch` çağrısı
yapılarak kontrol edildi, hepsi düzeltildi (metin içi atıflar + Kaynakça). Not: iki farklı kaynağın
ilk yazarı da "Tian" soyadını taşıyor (Katherine Tian 2023, Zailong Tian 2025) — metin içi atıflar
karışmasın diye "K. Tian" / "Z. Tian" olarak ayrıştırıldı.

### Açık bulgu: en yakın önceki çalışma PREREG'in "kesişim boş" çerçevesini nüanslandırıyor
Proskurina vd. (NAACL 2024 Findings, arXiv:2405.00632) GPTQ 4-bit kuantizasyonun LLM güvenini
**ECE ve ACE dahil** kalibrasyon metrikleriyle, altı benchmark'ta (ArcEasy, BoolQ, HellaSwag,
OpenBookQA, PiQA, XStory) ölçüyor. `PREREG.md` §1'in "LLM-kalibrasyon tarafı kuantizasyona hiç
bakmıyor" cümlesi literal okunduğunda tam doğru değil. **`PREREG.md`'ye dokunulmadı** (donmuş,
PROTOKOL Kural 4/7) — bu bir tasarım sapması değil (hiçbir kuantizasyon parametresi/koşul/metrik
değişmedi), yalnızca §1'deki arka plan çerçevesinin literatürle tam örtüşmediği bir nüans.
`YAPILACAKLAR.md`'ye açık bulgu olarak kaydedildi, `paper.md` §2.2'de kaynak dürüstçe atıfla
verilip bu çalışmanın asıl farkı (tam bit merdiveni + mod karşılaştırması + mixed recipe + MLX
runtime + ön-kayıtlı falsifikasyon) netleştirildi — Proskurina vd.'nin kapsamadığı beş somut
metodolojik boşluk maddelenerek.

### Yakalanan hata: kendi taslağımdaki bir elle-sayma hatası
Tartışma taslağını yazarken H2 (yön hipotezi) için confirming/contradicting hücre sayılarını
`verdicts.json`'daki listeleri elle sayarak "13 confirming / 32 contradicting" yazmıştım. Tabloları
ve metni bitirdikten sonra, PROTOKOL Kural 5 gereği kendi iddiamı koda karşı kontrol ederken
(`python -c "... Counter(...)"`), gerçek sayının **13 confirming / 36 contradicting** olduğu
ortaya çıktı (36 satırlık listeyi elle sayarken 4 satır atlamışım). Model başına dağılım da
(`llama3.2-3b`: 10 confirm/2 contradict, `qwen2.5-1.5b`: 3 confirm/16 contradict, `qwen2.5-3b`:
0 confirm/18 contradict) programatik olarak yeniden hesaplanıp metne işlendi. Diğer tüm sayısal
iddialar (H1 monotonluk, H3 örtüşme, H4 falsifikasyon, floor-control ECE aralığı, effective_bits)
aynı yöntemle (`pandas`/`json` ile CSV'lerden doğrudan okuma, elle transkripsiyon değil) üretildiği
için ayrıca hata bulunmadı.

### Fonksiyonel/varlık doğrulaması
`paper.md`'nin gömdüğü 3 figür yolu (`results/figures/figure{1,2,3}_*.png`) `ls` ile gerçekten var
olduğu doğrulandı. `git diff HEAD -- tests/` boş. Kabul kriteri komutu YAPILACAKLAR'da tanımlı
değildi (yalnızca içerik gereksinimi listesi).

---

## Görev 14 arası — Uçtan uca denetim (2026-07-26, kullanıcı talebiyle)

Kullanıcı, release/DOI öncesi her şeyin "best in class" ve çelişkisiz olduğunun uçtan uca
denetlenmesini istedi. 15 maddelik denetim yapıldı; kanıtlı sonuçlar:

**Temiz çıkanlar:** PREREG donmuş (`git diff c5ea71c..HEAD -- PREREG.md` boş); 62/62 test;
prompt SHA eşleşiyor; 114/114 hücre ok; README+paper'daki TÜM sayılar programatik doğrulandı
(paper Tablo 1: 42 hücre, Tablo 2: 12 satır, Tablo 3: 6 satır, Tablo 4: 24 satır — 0 hata;
README ECE+accuracy değerleri CSV/parquet'ten birebir; H2 metin içi sayılar 13/36, 10/2, 16,
"every evaluated cell"=0 conf/18 cont verdicts.json'la birebir); kaynakça 3/6 spot-check
(başlık+tam yazar listesi arXiv'den) birebir; SPEC↔src imzaları tam, Changelog 5 girdiyle
eksiksiz; DEVIATIONS boş ve paper §7 ile tutarlı; make verify canlı geçti; secret taraması temiz.

**Bulunan ve düzeltilen:**
1. **PROTOKOL Kural 8 ihlali:** 3 push-edilmemiş commit'te `Co-Authored-By: Claude` trailer'ı.
   Kullanıcı onayıyla, Kural 8'deki emsalle aynı yöntemle temizlendi: `filter-branch --msg-filter`
   yalnızca origin/main..HEAD (push edilmemiş 20 commit) aralığında; **içerik diff'i boş
   doğrulandı** (`git diff backup-pre-trailer-clean HEAD --stat` → boş), commit sayısı 20→20,
   PREREG commit'ine (push'lu) dokunulmadı. Yedek branch: `backup-pre-trailer-clean`.
2. **README H1 eksik anlatımı:** ihlal her iki benchmark'ta (verdicts+paper doğru veriyordu),
   README yalnızca ARC-Challenge diyordu — MMLU sayıları (0.422/0.292, Tablo 1'den) eklendi.
3. **`git clone <this repo>` placeholder'ı** (README+paper) → gerçek URL.
4. **Sibling study yerel path'i** (README+paper) → https://github.com/mertdurukan/imbalance-calibration
   (public olduğu `gh repo view` ile doğrulandı).
5. **LICENSE yoktu** → MIT eklendi (kullanıcı seçimi; Zenodo lisans ister, lisanssız public
   repo yasal olarak kullanılamaz).
6. **CITATION.cff yoktu** → eklendi (Zenodo metadata + GitHub cite kutusu bunu okur).
7. **Veri yayını kararı (kullanıcı onaylı):** `results/cells/` (5.2M parquet) + `results/meta/`
   (456K) .gitignore'dan çıkarılıp commit'lendi — paper §8'in "her sayı parquet'lerden yeniden
   hesaplanır" iddiası artık okuyucu için gerçek (6.6 saatlik grid koşusu olmadan ~10 dk'da
   analiz reprodüksiyonu).
8. **Tablo 4 caption netliği:** `within range` hükmünün CI-tabanlı olduğu (nokta 0.245 <
   taban 0.247 iken "yes" görünen satırın neden ihlal olmadığı) caption'a eklendi.

**Denetimde bulunup release'i bekleyen:** origin/main 20 commit gerideydi (public repo'da
kod/sonuç/paper görünmüyordu) — push, release adımıyla birlikte yapılacak.

---

## Görev 14 (devam) — arXiv paketi (2026-07-26)

### Karar: `paper.md` tek kaynak kalır, arXiv sürümü **türetilir**

İki seçenek vardı: (a) arXiv için ayrı bir `paper-arxiv.md`/`.tex` tutmak, (b) `paper.md`'yi tek
kaynak bırakıp LaTeX'i script'le üretmek. (a) seçildiğinde iki dosya kaçınılmaz olarak birbirinden
ayrışır — bir sayı düzeltmesi birinde yapılıp diğerinde yapılmaz ve hangisinin doğru olduğu
belirsizleşir; bu, PROTOKOL Kural 11'in ("üretilmiş çıktı elle düzenlenmez") tam olarak önlemeye
çalıştığı hata sınıfı, sadece tablolar yerine makale üzerinde. **(b) seçildi:**
`scripts/build_paper.py` + `make paper`, çıktı `build/arxiv/` (gitignore'da, üretilmiş artefakt).

### Açık bulgu: `paper.md`'de Zenodo DOI ve mutlak bağlantı yoktu

Paketi hazırlarken fark edildi (YAPILACAKLAR AÇIK BULGULAR, 2026-07-26): başlık bloğunda
"**Repository:** this repository." yazıyordu ve `PREREG.md`/`SPEC.md`/`README.md`/`GECMIS.md`'ye
6 **göreli** bağlantı vardı. Repo içinde doğru, ama arXiv'e giden tek başına duran bir PDF'te
ölü bağlantı — okuyucu ön-kaydı bulamaz, ki bu çalışmanın ana iddiası ön-kayıtlı olması.
Düzeltildi: mutlak GitHub URL'leri + concept DOI (`10.5281/zenodo.21596523`) başlık bloğunda ve
§8'de.

§8'e eklenen "arşiv `results/cells/` + `results/meta/` içeriyor, dolayısıyla analiz 6.6 saatlik
ölçüm ızgarasını yeniden koşmadan reprodüklenebilir" iddiası **varsayılmadı, doğrulandı:**
`git ls-tree -r --name-only v1.0.0 | grep -c "^results/cells/"` → 114, meta → 114. (Tag yerelde
yoktu, `git fetch --tags` ile çekildi — release `gh release create` ile GitHub tarafında
oluşturulmuştu.) DOI'ler de Zenodo API'sinden doğrulandı: sürüm `…524`, concept `…523`, başlık ve
MIT lisansı doğru.

### Kural 1 uygulaması: arXiv gereksinimleri bellekten değil kaynağından

Üç şey bellekten varsayılabilirdi, hepsi kaynağından okundu (WebFetch, 2026-07-26):

1. **PDF-mi-LaTeX-mi:** arXiv "does not accept dvi, PS, or PDF created from TeX/LaTeX source"
   (info.arxiv.org/help/submit). Yani yerel PDF gönderilemez — LaTeX kaynağı gönderilir, arXiv
   derler. Bu, DURUM.md'nin önceki oturumda yazdığı "markdown değil PDF gerekir" notunu
   **düzeltiyor**: gereken şey PDF değil, TeX kaynağı.
2. **Motor desteği:** xelatex Kasım 2025'te eklenmiş, lualatex **desteklenmiyor**; TeX Live 2023
   ve 2025 (2025 varsayılan) (info.arxiv.org/help/faq/texlive.html). Buna rağmen **pdflatex
   seçildi** — `paper.md`'de yalnızca 10 çeşit ASCII-dışı karakter var (—, §, ×, –, ö, Δ, ·, ≥,
   ≤, ć) ve bunları `newunicodechar` ile eşlemek, fontspec/font-varlığı riskini arXiv'in
   derleyicisine taşımaktan daha güvenli.
3. **Endorsement:** ilk gönderimde gerekiyor; kurumsal e-posta + sahiplenilmiş ortak-yazarlık
   olmadan yerleşik bir yazardan istenmesi gerekiyor (info.arxiv.org/help/endorsement).
   Kullanıcının adresi kurumsal değil → cs.LG için endorsement beklenmeli. ARXIV.md §2.

### Kural 6 uygulaması: "derlendi" değil, "arXiv'in yapacağı şeyi yaptım"

`main.tex` üretmek yeterli kanıt değildi. Sırayla:

- TinyTeX kuruldu (`--no-path` ile — installer aksi hâlde `/etc/paths.d` için sudo istiyor),
  eksik paketler tlmgr ile eklendi. İlk iki derleme denemesi gerçekten patladı
  (`caption.sty` yok → sonra `newunicodechar.sty` yok) — paket listesi tahminle değil hata
  mesajıyla oluştu.
- Derleme logu denetlendi: 4 `Overfull \hbox` bulundu, biri 94pt (uzun `\texttt{}` dosya yolu
  satır sonuna sığmıyor, `hyphenat[htt]` + `emergencystretch` ile 0.11pt'a indirildi —
  kalan 3 taşma tablo hizalamasında ve görsel olarak fark edilmez).
- PDF **gözle** okundu (9 sayfa, `pdftoppm` üzerinden). Üç gerçek kusur bulundu, hiçbiri log'da
  görünmüyordu: (a) her figür caption'ı "Figure 1: Figure 1: …" diye çift numaralanıyordu
  (markdown alt-text'i kendi numarasını taşıyor, LaTeX bir de kendisi ekliyor); (b) Figure 2
  neredeyse tamamen boş bir sayfada yüzüyordu (float yerleşimi → `!ht`); (c) abstract'teki URL
  `xurl` yüzünden "ht tps://…" diye kelime ortasından kırılıyordu (autolink → etiketli link).
  Üçü de düzeltildi, sayfa sayısı 10 → 9.
- **Dönüşümün sadakati programatik doğrulandı:** `paper.md`'nin tablo satırlarındaki 237 tekil
  ondalık sayının ve 6 arXiv künyesinin hepsi `pdftotext` çıktısında bulundu (eksik: 0). Pandoc'un
  sayıları bozmadığı bir iddia değil, sayım.
- **Tarball tek başına, temiz bir dizinde derlendi** — arXiv'in yapacağı işlem. `main.tex` + 3 PNG
  açıldı, iki geçiş pdflatex koşuldu, birebir aynı 9 sayfa/768054 bayt çıktı. Repo dizinine gizli
  bir bağımlılık kalmadığının kanıtı.

### Karar: yayın araçları `requirements.txt`'e eklenmedi

`pandoc`, TinyTeX ve `poppler` araştırma bağımlılığı değil, yazım/yayın aracı. SPEC §0 madde 8
sabitlenmiş **Python** bağımlılıklarını düzenliyor ve `make reproduce` bunların hiçbirine ihtiyaç
duymuyor — `make paper` ayrı, opsiyonel bir hedef. Kurulum komutları ARXIV.md §6'da kayıtlı.

### Gönderilmeyen şey

arXiv'e **gönderim yapılmadı.** Ajanın kullanıcının arXiv hesabına girmesi doğru değil ve
gönderim geri alınamaz bir dış eylem (duyurulduktan sonra kaldırılamaz, yalnızca v2 eklenebilir).
Paket hazır, metadata (kategori, comments, lisans, abstract) ARXIV.md §3'te forma girilecek
biçimde yazılı, adımlar §4'te. Kullanıcının kendi yapacağı iş: endorsement + form + arXiv'in
derlediği PDF'i yerel `main.pdf` ile karşılaştırıp onaylama.

### Bayat kaynak hatası: arXiv endorsement politikası (2026-07-26)

`ARXIV.md`'nin ilk yazımında endorsement bölümü arXiv'in **statik yardım sayfasından**
(info.arxiv.org/help/endorsement) yazıldı ve "kurumsal e-posta + ortak-yazarlık sahiplenme →
otomatik onay" izlenimi verdi. Kullanıcı gönderim adresini sorunca (gmail yerine
`mert@durukan.dev`) konu yeniden arandı ve arXiv'in **21 Ocak 2026 tarihli blog duyurusu**
bulundu: otomatik endorsement artık akademik e-posta **ve** önceden arXiv yazarlığının ikisini
birden istiyor (Aralık 2025'te Matematik'te başlayıp genelleştirilmiş; arXiv'in gerekçesi
"bilimsel olmayan gönderimlerde sürdürülemez artış"). Yani:

- E-posta adresi seçimi endorsement'ı **hiç** değiştirmiyor — `durukan.dev` (Google Workspace,
  kişisel alan adı) de bir gmail adresi de akademik kurum sayılmıyor, ve ikinci koşul (önceden
  arXiv yazarlığı) zaten sağlanmıyor.
- Tek yol **kişisel endorsement**: cs alanında yerleşik bir arXiv yazarından. arXiv personeli
  endorsement veremiyor.

**Ders:** PROTOKOL Kural 1'in tazelik maddesi, bir kurumun **kendi** dokümantasyonu için de
geçerli — yardım sayfası, o kurumun politika duyurusundan bayat olabilir. Bu çalışmanın konu
seçimi de tam aynı hata sınıfından kurtarılmıştı (bkz. "Neden bu konu"): Ocak 2026 bilgisiyle
açık görünen bir çelişki, Temmuz'da doğrulandığında çözülmüştü. Aynı ay içinde ikinci kez.

### Karar: yazışma adresi `mert@durukan.dev`, git adresi değişmedi

Repoda üç ayrı adres dolaşıyordu: commit'lerin hepsi `1mertacc@gmail.com` ile imzalı, Claude
oturum bağlamında `costorymind@gmail.com` görünüyor (ve ben ARXIV.md'ye onu yazmıştım — repoda
başka hiçbir yerde geçmiyordu), kullanıcı üçüncü olarak `mert@durukan.dev`'i önerdi. Adresin
gerçekten mail alabildiği varsayılmadı, DNS'ten doğrulandı: `MX 1 smtp.google.com` +
`v=spf1 include:_spf.google.com` → çalışan bir Google Workspace alan adı.

Kullanıcı kararı: `mert@durukan.dev` **hem** `paper.md` başlık bloğuna **hem** `CITATION.cff`'e
(akademik norm; karşılığında adres arXiv PDF'i, GitHub ve Zenodo üzerinden kalıcı olarak
taranabilir hale geliyor — bu ödünleşim kullanıcıya açıkça soruldu). Git commit adresi
**değişmedi**: tüm geçmiş `1mertacc@gmail.com` ile kalıyor, tek tutarlı yazar kimliği ve GitHub
atfı korunuyor; geçmiş zaten yeniden yazılmazdı (Kural 8).

**Yakalanan tuzak:** adresi yalnızca `paper.md`'ye yazmak yetmiyordu. `build_paper.py` başlık
bloğunu LaTeX gövdesine **hiç aktarmıyor** (yalnızca "Code and data" cümlesini abstract'e
taşıyor), yani yazışma adresi arXiv PDF'inde **sessizce görünmeyecekti** — repo kopyasında
görünüp yayınlanan kopyada kaybolan bir fark. `author_line()` eklendi: adres başlık bloğundan
okunup `\author{}` satırına mailto linkiyle taşınıyor. PDF'te doğrulandı (`pdftotext | grep` → 1
eşleşme, sayfa 1 gözle kontrol edildi) ve tarball yeniden temiz dizinde derlendi.

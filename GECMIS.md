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

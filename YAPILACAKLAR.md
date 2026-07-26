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

### [x] Görev 9 — Tam koşu — TAMAMLANDI 2026-07-26
**Ön koşul:** Görev 8 elle onaylandı
**Yap:** `caffeinate -i make reproduce`. Kesintiye dayanıklı olmalı — başladıktan 2 dk sonra
`Ctrl+C` yapıp tekrar başlat, tamamlanmış hücreleri atladığını **doğrula**.

**Kabul kriteri:** hücre sayısı (uygunluk kapısına göre değişken — bkz. not), `status` dağılımı,
her farklı hatanın **tam metni**. Başarısız hücre varsa `DEVIATIONS.md`'ye açıklaması yazılır —
**silinmez.**

**Sonuçlara bakma.** Bu görevin çıktısı sağlık raporudur, metrik ortalaması değil.

**Sonuç:** Kesinti/devam testi geçti (bkz. GECMIS.md — ~5 dk sonra kasten kesildi, yeniden
başlatıldığında tamamlanmış hücrelerin `mtime`'ı değişmedi). İlk koşu **88/88 hücre `status="ok"`**
verdi (140 değil — o koşuda Llama modelleri, sonradan bulunan bir ölçüm hatası yüzünden
`eligible=false` çıkmıştı, sadece bf16'ları çalıştı). Koşu sürerken canlı izlerken kritik bir hata
bulundu ve düzeltildi (bkz. AÇIK BULGULAR, "Llama-3.2 BOS token hatası"). Düzeltmeden sonra 4 eski
llama bf16 hücresi silinip `python -m src.runner` tekrar çalıştırıldı (Qwen'in 84 hücresi cache'ten
atlandı, yalnızca llama yeniden ölçüldü) — **nihai: 114/114 hücre `status="ok"`, hiç `failed` yok.**
Nihai `eligibility.json`: `llama3.2-1b` hâlâ eşiğin altında (arc 0.475/mmlu 0.435, eligible=false,
2 hücre — yalnızca bf16), ama **`llama3.2-3b` eşiği geçti** (arc 0.713/mmlu 0.569, eligible=true) —
düzeltmeden önce bu modelin tamamen elenmiş olacağı bir sonuçtu, tam 14-koşullu merdiveni işlendi
(28 hücre). Model başına: `qwen2.5-0.5b` 28, `qwen2.5-1.5b` 28, `qwen2.5-3b` 28, `llama3.2-3b` 28,
`llama3.2-1b` 2. Hiç başarısız hücre yok, `DEVIATIONS.md`'ye eklenecek bir şey yok.

---

## FAZ 5 — Analiz

### [x] Görev 10 — `src/analyze.py` — sadece ön-kayıtlı tablolar — TAMAMLANDI 2026-07-26
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

**Sonuç:** SPEC §3'e altı saf verdict fonksiyonu eklendi (`_intervals_overlap`,
`_paired_bootstrap_delta`, `_h1_ladder_verdict`, `_h2_direction_verdict`, `_h3_mode_verdict`,
`_h4_recipe_verdict`) — test-önce + kullanıcı onayı (iki turda, H2 ilk turda eksikti) +
mutasyon kanıtı (`_intervals_overlap`'in sınır karşılaştırması bozuldu, yalnızca ilgili test
doğru sebeple FAIL etti). `pytest tests/` → 62 passed (27 yeni). Gerçek 114 hücrelik veri
üzerinde iki kez koşuldu (~9.5 dk/koşu): ilk koşu `overconfidence_rate`'in bazı aşırı-sıkıştırma
hücrelerinde tanımsız (0/0) olduğunu ortaya çıkardı (kod hatası değil — `qwen2.5-3b`'nin 2-bit
hücrelerinde hiçbir tahmin %90 güveni aşmıyor), `overconfidence_rate_n_qualifying` kolonuyla
şeffaflaştırıldı, ikinci koşu temiz çıktı. Tablo 1-4 + `table5_floor_control.csv` (PREREG §4.2
zorunlu ayrı ifşa) + `verdicts.json` (H1-H4 PASS/FAIL) + 3 figür üretildi, elle incelendi,
anormallik yok. `qwen2.5-3b`'nin 3-bit hücresinde H1'i gerçekten ihlal eden (mutasyonlu test
değil, gerçek veri) istatistiksel olarak anlamlı bir ters-U bulundu — Görev 13'e not, bkz.
GECMIS.md "Görev 10". `matplotlib==3.11.1` eklendi (`requirements.txt`/`requirements.lock.txt`).

---

### [x] Görev 11 — Veri tarama yasağı kontrolü — TAMAMLANDI 2026-07-26
**Ön koşul:** Görev 10
**Yap:** Analiz çıktısını gözden geçir: ön-kayıtta olmayan hiçbir test yapılmamış mı?
Yapıldıysa "Exploratory" etiketi var mı? Yoksa kaldır veya etiketle.

**Sonuç:** `src/analyze.py`, `results/tables/*.csv`, `results/figures/*.png` ve
`tests/test_analyze.py` elle incelendi. Tablo 1-4 PREREG §5/§4.5'teki metriklerle 1:1
örtüşüyor (H1: ECE/slope/intercept/Brier; H2: ortalama güven + aşırı-güven oranı; H3/H4:
yalnızca ECE). `table5_floor_control.csv` PREREG §4.2'nin **zorunlu ayrı ifşası** —
`verdicts.json`'a girmiyor, beşinci bir hipotez tablosu değil; içindeki `accuracy` kolonu
§4.5'in "reported but not the estimand" listesinde açıkça izinli. `overconfidence_rate_n_qualifying`
(Görev 10'da eklenen) yeni bir istatistik değil, var olan bir metriğin NaN'ını açıklayan
şeffaflık kolonu. Üç figür PREREG §5'teki tanımlarla birebir. `tests/test_analyze.py`'deki
27 test yalnızca H1-H4'ün altı verdict fonksiyonunu sınıyor, kayıt dışı hiçbir karşılaştırma
yok. Kodun hiçbir yerinde "Exploratory" bölümü yok — çünkü hesaplanmış kayıt-dışı hiçbir şey
yok, dolayısıyla etiketlenecek/kaldırılacak bir şey bulunmadı. **Değişiklik yapılmadı.**

---

## FAZ 6 — Yazı + dağıtım

### [x] Görev 12 — `README.md` — TAMAMLANDI 2026-07-26
Repo ön yüzü: tek paragraf bulgu, manşet sayılar, `make reproduce`, sınırlar, abartı yok.

**Sonuç:** `README.md` yazıldı (95 satır). Tek paragraf bulgu: H1'in monoton olmayan gerçek
ihlali (qwen2.5-3b/arc, 3-bit ECE 0.434 > 2-bit ECE 0.254, CI'lar örtüşmüyor) + H2'nin model
başına yön tutarsızlığı (llama3.2-3b onaylıyor, qwen2.5-1.5b çoğunlukla çelişiyor) + H3'ün
6 hücreden yalnızca 1'inde fark göstermesi — hepsi `verdicts.json`/`table1`/`table3`'ten
**birebir okunarak** yazıldı, hesaplama uydurulmadı. Manşet tablosundaki 2-bit doğruluk
sayıları (0.244/0.268/0.224) ilk taslakta tahmindi, gerçek parquet'lerden
(`df['is_correct'].mean()`) hesaplanarak düzeltildi. Sızıntı/kapsam: "sibling study" için
uydurma bir GitHub URL'i yazılmıştı (kural: URL uydurma yasak) — fark edilip yerel yola
(`~/github-projects/imbalance-calibration`) düzeltildi. Kabul kriteri komutu tanımlı değildi
(görev metni yalnızca içerik gereksinimi listeliyor); kanıt: dosya var (95 satır),
`git diff HEAD -- tests/` boş.

### [x] Görev 13 — `paper.md` — TAMAMLANDI 2026-07-26
Abstract, giriş, related work (**kaynaklar tam metin okunacak**, ≤15 kelime alıntı), yöntem,
sonuçlar (gömülü tablo+figür), tartışma, sınırlar, reprodüksiyon.

**Sonuç:** `paper.md` yazıldı (471 satır, 8 bölüm). Related work için 6 gerçek kaynak WebSearch
ile bulundu, her biri WebFetch ile (abstract sayfası + `2405.00632` için ayrıca HTML tam metin)
okunarak doğrulandı — yazar isimleri de ayrıca doğrulandı (ilk taslakta 4 kaynağın yazar adını
yanlış tahmin etmiştim: "Kwon"→gerçeği Proskurina, "Bhat & Chen"→gerçeği Rajesh vd., "Xiong"→
gerçeği K. Tian, "Zhao"→gerçeği Zhou; hepsi WebFetch ile gerçek arXiv sayfasından alınan isimlerle
düzeltildi). Her alıntı ≤15 kelime ve kaynağın gerçek metninden birebir substring (programatik
kelime sayımıyla doğrulandı). **Önemli bulgu:** en yakın önceki çalışma (Proskurina vd., NAACL
2024, arXiv:2405.00632) GPTQ 4-bit kuantizasyonun LLM güvenini ECE/ACE ile ölçüyor — PREREG §1'in
"kesişim boş" çerçevesini kısmen nüanslandırıyor, AÇIK BULGULAR'a kaydedildi ve makalede §2.2'de
dürüstçe ele alındı (PREREG'e dokunulmadı, donmuş). Sonuçlar bölümündeki tüm sayılar (Tablo 1-5,
verdicts.json) doğrudan `results/tables/*.csv` + `verdicts.json`'dan pandas/python ile okunarak
üretildi, elle yazılmadı — bir istisna dışında: H2 özet metnindeki confirming/contradicting hücre
sayılarını ilk yazımda elle sayarken hata yaptım (13/32 yazmıştım), `verdicts.json`'u
`Counter`'la programatik sayarak gerçek sayının 13/36 olduğunu buldum ve düzeltim (PROTOKOL
Kural 5 — iddiaya değil koda bak, kendi taslağıma bile). 5 tablo + 3 figür (`results/figures/`'da
gerçekten var, embed edilen path'ler doğrulandı) makaleye gömüldü. Kabul kriteri komutu tanımlı
değildi (yalnızca içerik gereksinimi); kanıt: dosya var, `git diff HEAD -- tests/` boş.

### [ ] Görev 14 — Dağıtım
GitHub vitrini (description, topics), release + Zenodo DOI, sonra arXiv / doğrudan e-posta.

**Kısmi ilerleme (2026-07-26):** GitHub vitrini tamamlandı — `gh repo edit` ile description
("Pre-registered study of how MLX quantization (bf16 → 2-bit, mxfp4, mixed recipes) breaks LLM
confidence calibration — not just accuracy.") ve 9 topic (`mlx`, `quantization`, `llm`,
`calibration`, `confidence-calibration`, `apple-silicon`, `machine-learning`,
`pre-registered-research`, `expected-calibration-error`) ayarlandı, `gh repo view` ile doğrulandı.
Kullanıcıya adım-adım onay protokolü soruldu (kullanıcı "adım adım, her adımda onay" seçti).

**Devam (2026-07-26, aynı gün ilerleyen saatler):** kullanıcı onayıyla repo **public** yapıldı
(`gh repo edit --visibility public`, secret ön-taraması temiz çıktıktan sonra). Kullanıcı
Zenodo'da GitHub entegrasyon toggle'ını açtı (release yayınlanınca DOI otomatik). Release
taslağı hazırken kullanıcı **uçtan uca denetim** istedi — 15 maddelik denetim yapıldı, 8 bulgu
düzeltildi (Kural 8 trailer temizliği içerik-diff-boş kanıtıyla, README H1 eksik anlatımı,
placeholder URL'ler, sibling gerçek URL, MIT LICENSE, CITATION.cff, cells+meta veri yayını,
Tablo 4 caption netliği — detay: GECMIS.md "Görev 14 arası"). **Kalan:** `git push` + release
v1.0.0 (taslak hazır, onay bekliyor) → Zenodo DOI → DOI rozeti README'ye → arXiv/e-posta
(ayrı karar).

**Devam (2026-07-26, push + release + DOI):** `git push origin main` temiz gitti (fast-forward),
`gh release create v1.0.0` yayınlandı, Zenodo entegrasyonu DOI'yi otomatik bastı — sürüm DOI
`10.5281/zenodo.21596524`, concept DOI `10.5281/zenodo.21596523`, API'den doğrulandı (başlık,
sürüm, MIT lisansı). Concept-DOI rozeti README'ye, `doi:` alanı CITATION.cff'e eklendi.

**Devam (2026-07-26, arXiv paketi):** `scripts/build_paper.py` + `make paper` yazıldı —
`paper.md` **tek kaynak** kalıyor, arXiv LaTeX'i ondan türetiliyor (ayrı bir makale sürümü
tutulmuyor: sürüm sapması riski, bkz. GECMIS.md). Çıktı `build/arxiv/` (gitignore'da):
`arxiv-submission.tar.gz` (`main.tex` + 3 figür, 408 KB — arXiv'e yüklenecek dosya),
`main.pdf` (9 sayfa, yalnızca okuma için), `abstract-for-arxiv-form.txt` (düz-ASCII, 233 kelime).
`ARXIV.md` yazıldı: forma girilecek metadata (cs.LG primary + cs.CL cross-list, comments,
lisans), adım adım gönderim, endorsement gereksinimi, artık riskler.

arXiv gereksinimleri kaynağından doğrulandı (Kural 1) ve önceki oturumun bir notu **düzeltildi**:
arXiv TeX'ten üretilmiş PDF'i **kabul etmiyor** — LaTeX kaynağı gönderilir, arXiv kendisi derler.
`paper.md`'ye Zenodo DOI + mutlak URL'ler eklendi (açık bulgu, yukarıda). Kural 6 doğrulaması:
PDF gözle okundu ve log'da görünmeyen üç kusur bulunup düzeltildi (çift figür numarası, boş
sayfada yüzen Figure 2, abstract'te kelime ortasından kırılan URL — 10→9 sayfa); dönüşümün
sadakati programatik sayıldı (237 tekil tablo sayısı + 6 arXiv künyesi, eksik 0); tarball **temiz
bir dizinde tek başına derlendi** (arXiv'in yapacağı işlem, birebir aynı çıktı).
`pytest tests/` → 62 passed, `git diff HEAD -- tests/` boş.

**Kalan (kullanıcıda, delege edilemez):** arXiv gönderimi. Ajan kullanıcının arXiv hesabına
girmedi — gönderim geri alınamaz bir dış eylem (duyurulduktan sonra kaldırılamaz, yalnızca v2).
Sıra: endorsement → form → tarball → arXiv'in derlediği PDF'i yerel `main.pdf` ile karşılaştır →
onayla. Bkz. `ARXIV.md`.

---

## 🔍 AÇIK BULGULAR

### ⚠️ 2026-07-26 BAĞIMSIZ DENETİM (PROTOKOL Kural 10) — kritik bulgular

> Kural 10 ("kodu yazan bağlam o kodu onaylayamaz") bu projede ilk kez gerçekten işletildi:
> üç **sıfır bağlamlı** alt-ajan, yalnızca dosyalar + sözleşme verilerek, yazarın gerekçeleri
> verilmeden çalıştırıldı. Aşağıdaki bulguların **hepsi bu bağlamda bağımsız olarak yeniden
> doğrulandı** (Kural 5: ajanın iddiası kanıt değil) — komutlar ve çıktılar oturum kaydında.
> Denetim ayrıca ~446 sayısal iddiayı **doğru** buldu (tüm tablo hücreleri CSV'lerle birebir,
> 16 değer parquet'ten bağımsız yeniden hesaplandı) — kusurlar aritmetikte değil, **karar
> kurallarında ve düzyazı iddialarında.**

- [x] **[2026-07-26] KRİTİK — Abstract "dört hipotezin dördü de çürütüldü" diyor, ama H3 GEÇTİ.** `results/tables/verdicts.json` → `H3.overall_pass = True` (doğrulandı). Makalenin kendi §4.3'ü doğru yazıyor ("**Not falsified — but only just**", paper.md:276) ve §5'te "H3 survives only because one cell out of six carries it" diyor — ama abstract (paper.md:23), giriş (paper.md:67) ve §5'in açılışı (paper.md:369) "all four are falsified" diyor. Yani makalenin **en çok okunan cümlesi**, çalışmanın kendi mekanik hüküm dosyasının reddettiği bir çürütme iddia ediyor. Doğrusu: **4'te 3.** "Dördü de çürütüldü" daha temiz ve daha yayınlanabilir bir iddia — tam bu yüzden tehlikeli. · Engellediği görev: 14 (arXiv'e bu hâliyle gönderilmemeli) · Ciddiyet: **kritik**

- [x] **[2026-07-26] KRİTİK — H1'in karar kuralı ön-kayıtta yazılandan iki ayrı şekilde daha zayıf uygulanmış.** `PREREG.md:73`: "*Falsified if* the ECE **ordering across {8, 6, 5, 4, 3, 2}** is not monotone increasing, with 95% intervals excluding the reversal being noise." Bir kümenin üzerindeki *sıralama* iddiası **tüm çiftler** hakkındadır. `src/analyze.py:74-79` (`_h1_ladder_verdict`) ise `zip(rows, rows[1:])` ile **yalnızca komşu** bit-genişliği çiftlerini sınıyor. Tüm çiftler test edildiğinde **iki ek ihlal** ortaya çıkıyor (bu bağlamda yeniden hesaplandı): `qwen2.5-1.5b/mmlu` **8→3** (0.211 [0.185,0.240] vs 0.148 [0.127,0.180], aralıklar ayrık — komşu testi bu hücrede hiçbir şey bulmuyor) ve `qwen2.5-3b/mmlu` **8→4** (0.249 [0.223,0.277] vs 0.192 [0.166,0.220], ayrık). Sonuçları: (a) manşet sayı **6'da 2 değil 6'da 3** olmalı; (b) abstract'in "**one** model (Qwen2.5-3B) shows a … non-monotonic spike" ve §5'in "model-specific and localized (one model)" ifadeleri **iki modele** çıkıyor — bu makalenin merkezî yorumsal iddiası; (c) `verdicts.json` `qwen2.5-1.5b__mmlu` için `"monotone": true, "violations": []` kaydediyor, ön-kayıtlı ifadeye göre yanlış. **İkinci zayıflatma:** `PREREG.md:179` "Primary contrasts are **paired within item**" diyor, ama H1 hükmü eşleştirilmiş delta yerine **marjinal** ECE aralıklarının örtüşmesine bakıyor (`table1`'de `delta_ece_*` kolonları zaten mevcut, kullanılmıyor). Örtüşme testi muhafazakârdır — yani hatanın yönü **çürütmeyi az tespit etmek**, iki bakımdan da. · Engellediği görev: 14 · Ciddiyet: **kritik**

- [x] **[2026-07-26] KRİTİK — Taban kontrolü modeli uygunluk kapısını GEÇTİ, ama `role` alanıyla dışarıda tutuldu; makale sonra onun doğruluğunu şans seviyesi diye tanıtıyor.** `PREREG.md:126`: "a model enters the main grid only if its bf16 reference accuracy **on the pre-registered sample** is **>= 50%** on at least one benchmark." `results/eligibility.json` → `qwen2.5-0.5b: {arc_challenge: 0.518, mmlu: 0.436, eligible: **true**, role: floor_control}` — kapıyı **geçti** (0.518 ≥ 0.50). Ana ızgaradan çıkaran şey kural değil, `src/analyze.py`'deki `role != "floor_control"` filtresi. Ayrıca `PREREG.md`'nin kendi içinde bir **tutarsızlığı** var: kural "ön-kayıtlı örneklem" diyor, ama 0.5B'yi eleme gerekçesi 30 item'lık fizibilite ölçümünden gelen %33. Kararın kendisi post-hoc değil (PREREG:135 0.5B'yi adıyla eliyor), ama makale bu çelişkiyi **yüzeye çıkarmak yerine yanlış bir doğruluk tasviriyle üstünü örtmüş:** paper.md:345 "Accuracy stays near or below chance throughout the ladder" — oysa aynı cümlenin parantezi 0.518 diyor ve `table5`'te 28 hücrenin **23'ü 0.30 üstü, 16'sı 0.40 üstü** (arc ortalaması 0.416, mmlu 0.370; şans 0.25, `n_options` tüm 114.000 item'da 4 — doğrulandı). Yalnızca 4 hücre ≤0.25. Bu, §4.5'in taban kontrolünden çıkardığı sonucu ("calibration is not meaningfully measurable near chance accuracy") desteksiz bırakıyor. Ve paper.md:414'ün "mekanik kapı post-hoc model seçmeye karşı korur" iddiası, kapının çıktısının geçersiz kılınmış olmasına dayanıyor. · Engellediği görev: 14 · Ciddiyet: **kritik**

- [x] **[2026-07-26] Warmup item'ları aslında SKORLANIYOR — ön-kayıt ve makale skorlanmadığını söylüyor.** `PREREG.md:165`: "The first 20 items of every cell are warm-up and are **discarded, not scored**." `paper.md:171`: "first 20 … warm-up and discarded, **never scored**" — ama aynı makale "1000 items each" diyor. `src/measure.py` ilk 20 item'ı bir kez koşup atıyor, **sonra 1000 item'ın tamamını (o 20 dahil) skorluyor.** Doğrulandı: her hücre parquet'i 1000 satır ve ilk 20 warmup item'ının **20/20'si** çıktıda mevcut. "1000 item" ile "ilk 20 asla skorlanmadı" birlikte doğru olamaz; ön-kayıtlı tasarım 980 skorlanmış item ima ediyor. Uygulama ön-kayıtın *gerekçesini* (ölçüm ısınmış durumda yapılmalı) muhtemelen daha iyi karşılıyor — 1000 item'ın hepsi ısınmış durumda ölçülüyor — ama bu bir **sapma** ve `DEVIATIONS.md`'ye yazılmamış. `SPEC.md:196` de yanlış tarif ediyor ("not scored, not returned, not counted"), `src/measure.py`'deki yorum satırı da ("never scored either way"). · Engellediği görev: 14 · Ciddiyet: orta

- [x] **[2026-07-26] Makalenin ana tabloları ön-kayıtlı estimand'ların bir kısmını GÖSTERMİYOR.** `PREREG.md:213` — "**Table 1 (H1):** ECE, slope, intercept, Brier per model x bit-width, **paired against bf16**, with 95% intervals." Makalenin Tablo 1'i **yalnızca ECE** gösteriyor ve bf16'ya eşleştirilmiş delta değil, ham hücre ECE'si veriyor. `PREREG.md:214` — "**Table 2 (H2):** mean confidence on correct vs incorrect, and **overconfidence rate**, per condition." Makalenin Tablo 2'si Δintercept ve Δconf(incorrect) gösteriyor; **mean confidence on correct** ve **overconfidence rate** makalede hiç yok. Hepsi CSV'lerde **mevcut** (doğrulandı: `table1` kolonlarında `slope/intercept/brier` + tüm `delta_*`; `table2`'de `mean_conf_correct`, `overconfidence_rate*`) — yani hesaplanmış, sonra gösterilmemiş. Ters yönde kaçak **yok** (denetim ayrıca doğruladı: kayıt dışı hiçbir estimand tablolara girmemiş) — kusur tek yönlü: **eksiklik.** Görev 11 ("veri tarama yasağı kontrolü") yalnızca fazlalık arıyordu, eksikliği aramıyordu. · Engellediği görev: 14 · Ciddiyet: orta

- [x] **[2026-07-26] `DEVIATIONS.md` boş ve makale §7 "None" diyor — ama en az dört sapma kayıt dışı.** Yukarıdaki dördü (uygunluk kuralının `role` ile geçersiz kılınması, H1 sıralama kriterinin komşu çiftlere daraltılması, warmup'ın skorlanması, ön-kayıtlı tablo içeriğinin raporlanmaması) tam olarak paper.md:422'nin "no design element (conditions, benchmarks, **protocol, metrics, eligibility rule**) was changed" diye saydığı kategorilere düşüyor. Boş bir sapma defteri, ön-kayıtlı bir makalenin verebileceği en güçlü bütünlük sinyali — ve burada boş olmasının nedeni sapma olmaması değil, **sapmaların fark edilmemesi.** Ayrıca kayıt dışı iki tanım kararı: `overconfidence_rate`'in %90 eşiği (`src/config.py`, PREREG "fraction of high-confidence errors" diyor, eşik belirtmiyor) ve H4'ün bileşen aralığının nokta tahminlerle mi aralıklarla mı sınırlandığı (PREREG belirtmiyor, kod nokta tahmin seçmiş — muhafazakâr taraf). · Engellediği görev: 14 · Ciddiyet: orta

- [x] **[2026-07-26] Beş adet, veriyle çelişen abartılı nicelik ifadesi (düzyazı).** Hepsi bu bağlamda yeniden hesaplandı: (1) paper.md:216 "ECE **nearly doubles**" — gerçek oranlar arc 3bit/4bit **3.49×**, 3bit/2bit **1.71×**, mmlu **2.19×** ve **1.45×**; dördün yalnızca biri "neredeyse iki kat", ikisi iki katı aşıyor, biri çok altında. (2) paper.md:165 "nominal 4-bit costs **4.501–4.502** bits/weight" — yalnızca `affine_b4_g64` için doğru; `affine_b4_g128` ve `mxfp4_b4_g32` 4.251–4.252, `affine_b4_g32` **5.001–5.002**. Gerçek aralık **4.251–5.002** ve g32 varyantının nominal 5-bit'ten pahalı olması yazılandan daha ilginç bir olgu. (3) paper.md:335 "recipes **interpolate almost exactly**" — 24 recipe'nin **9'unun** nokta tahmini bileşen aralığının dışında; Tablo 4 caption'ı örnek olarak dokuzun **en küçüğünü** seçmiş (0.245 vs 0.247, fark 0.002), en büyüğü 10 katı (`qwen2.5-1.5b`/mmlu/`mixed_3_6`: 0.128 vs taban 0.148). "1 of 24 falsified" hükmü CI kuralı altında doğru, ama "almost exactly" 9/24'ün gösterdiği şey değil. (4) paper.md:444 "**~6.6-hour** measurement grid" — gerçek `sum(wall_seconds)` **6.95 saat**, dönüşümle **7.11 saat**. (5) README.md:16 "ECE … worsens **sharply at 2–3 bits**" — 3-bit'te 6 hücrenin yalnızca 4'ü anlamlı kötüleşiyor; `qwen2.5-1.5b`/MMLU anlamlı şekilde **iyileşiyor** (0.213→0.148). · Engellediği görev: 14 · Ciddiyet: orta

- [x] **[2026-07-26] Abstract'te yanıltıcı bir yan yanalık ve iki küçük sayım hatası.** (a) paper.md:21-22 "three instruction-tuned models … across 14 quantization conditions and two … benchmarks … **114 cells total**" — 3×14×2 = **84** (doğrulandı: üç ana modelin hücre sayısı tam 84); 114'e çıkmak için taban kontrolü (28) ve `llama3.2-1b` (2) gerekiyor, ikisi de o cümlede anılmıyor. Aritmetik yanlış değil, ama cümle yanlış çıkarıma davet ediyor. (b) paper.md:353 "**two** of the extreme-compression cells" — sayı **4** (2 koşul × 2 benchmark; doğrulandı: `n_qualifying==0` olan tam bu 4 hücre). (c) "14 quantization conditions" — 13 kuantize + 1 bf16 referansı; §3 doğru yazıyor. (d) paper.md:66 "RQ1–RQ4 in PREREG **§2**" — §2 araştırma *soruları*, hipotezler §3'te H1–H4. (e) paper.md:401 "Stated in advance in PREREG §6" — altı sınırın beşi öyle, "Three eligible main models" PREREG §6'da yok (iyi bir sınır, sadece ön-kayıta atfedilmemeli). · Engellediği görev: 14 · Ciddiyet: düşük

- [x] **[2026-07-26] PREREG "exploratory olan ayrı bir bölümde etiketlenir" diyor; makale bunu Results'ın içine gömmüş.** `PREREG.md:223`: "Anything not listed above is exploratory and will be labelled as such **in a separate section**." paper.md:353-361 kayıt dışı belirsizlik/`n_qualifying` gözlemini §4.5'in içinde veriyor ve PREREG §5'e atıf yapıyor — kuralın yarısı. Gözlem dürüstçe "unregistered" diye işaretli ve hükümlere girmiyor, yani yanlış beyan değil **yapısal uyumsuzluk**. Görev 11 bunu kaçırdı çünkü yalnızca "kayıt dışı bir şey var mı" diye baktı, "doğru yerde mi" diye bakmadı. · Engellediği görev: 14 · Ciddiyet: düşük

- [x] **[2026-07-26]** Görev 13 related work araştırması sırasında (tam metin okundu, WebFetch ile) PREREG §1'in "kesişim boş" iddiasını kısmen nüanslandıran gerçek bir kaynak bulundu: Kwon vd., "When Quantization Affects Confidence of Large Language Models?" (NAACL 2024 Findings, arXiv:2405.00632) — GPTQ 4-bit kuantizasyonun LLM güvenini nasıl etkilediğini **ECE ve ACE dahil** kalibrasyon metrikleriyle ölçüyor (Mistral-7B/Llama-7B/560M, ArcEasy/BoolQ/HellaSwag/OpenBookQA/PiQA/XStory). PREREG'in "LLM-kalibrasyon tarafı kuantizasyona hiç bakmıyor" cümlesi tam doğru değil — ama bu çalışma tek bir bit-genişliği (4-bit), tek kuantizasyon modu (GPTQ, CUDA), tam bir bit merdiveni yok, mxfp4/mixed recipe yok, MLX yok, ön-kayıtlı falsifikasyon kriteri yok. **PREREG.md'ye dokunulmadı** (donmuş, kural 4) — bu bir tasarım sapması değil, arka plan çerçevesi nüansı; `paper.md`'nin Related Work bölümünde bu kaynak **en yakın önceki çalışma** olarak dürüstçe atıfla verilecek, "kesişim tamamen boş" iddiası yerine "MLX'e/tam bit merdivenine/ön-kayıtlı tasarıma özgü boşluk" olarak çerçevelenecek. Bilime/sonuçlara etkisi yok, yalnızca yazım çerçevesi. · Engellediği görev: yok (çözüldü, Görev 13 içinde ele alınacak) · Ciddiyet: düşük

- [x] **[2026-07-26]** `requirements.txt`'te **bildirilmemiş doğrudan bağımlılıklar** vardı: `src/` içindeki modüller `pandas` ve `numpy`'ı doğrudan import ediyor ve hücre başına parquet okuma/yazma `pyarrow` gerektiriyor, ama üçü de `requirements.txt`'te yoktu — yalnızca `datasets`/`statsmodels` üzerinden **dolaylı** geliyorlardı (`pip show pandas` → `Required-by: datasets, statsmodels`). Bugün çalışıyor, ama tesadüfen: `datasets` bir gün pandas'ı bırakırsa ya da farklı bir sürüm çözerse `make setup && make reproduce` zinciri kırılır ve bu, çalışmanın en temel iddiasının (tek komutla reprodüklenebilir) dayanağını ortadan kaldırır. SPEC §0 madde 8'in ruhuna da aykırı: doğrudan kullanılan bir şey sabitlenmiş biçimde bildirilmeli. **Çözüldü:** `numpy==2.5.1`, `pandas==3.0.5`, `pyarrow==25.0.0` — hepsi `requirements.lock.txt`'te **zaten var olan** sürümlerle — `requirements.txt`'e gerekçe yorumuyla eklendi. Değişiklik ortama sıfır etkili olduğu doğrulandı: `pip install --dry-run -r requirements.txt` hepsini "already satisfied" veriyor ve yeni `pip freeze` çıktısı `requirements.lock.txt` ile **birebir aynı** (diff boş). · Engellediği görev: yok · Ciddiyet: orta

> Çalışırken keşfedilen her eksik, belirsizlik, tutarsızlık **buraya yazılır** — sessizce
> geçilmez. Format:
> `- [ ] **[TARİH]** <bulgu> · Engellediği görev: <no|yok> · Ciddiyet: kritik|orta|düşük`

- [x] **[2026-07-26]** arXiv gönderim paketi hazırlanırken `paper.md`'de iki gerçek boşluk bulundu: (a) Zenodo DOI'si (`10.5281/zenodo.21596523`) hiçbir yerde geçmiyor — oysa DURUM.md arXiv ön koşulu olarak bunu listeliyor ve makale artık arşivlenmiş bir veri/kod sürümüne atıf verebilir durumda; (b) başlık bloğunda "**Repository:** this repository." yazıyor ve `PREREG.md`/`SPEC.md`/`README.md`/`GECMIS.md`'ye **göreli** bağlantılar var — repo içinde doğru çalışıyorlar ama arXiv'e giden tek başına duran PDF'te ölü bağlantıya dönüşüyorlar (okuyucu ön-kaydı bulamaz, ki bu çalışmanın ana iddiası "ön-kayıtlı" olması). Repo Temmuz 2026'da public olduğu için mutlak URL'ler artık yazılabilir. Çözüm: `paper.md` **tek kaynak** olarak kalır (ayrı bir arXiv sürümü tutulmaz — sürüm sapması riski), göreli bağlantılar mutlak GitHub URL'lerine çevrilir ve DOI eklenir; arXiv `.tex`'i bu tek kaynaktan pandoc ile üretilir. **Çözüldü (2026-07-26):** DOI başlık bloğuna ve §8'e eklendi (concept + sürüm DOI'si ayrı ayrı, Zenodo API'sinden doğrulanarak); 6 göreli `.md` bağlantısı mutlak GitHub URL'ine çevrildi; "this repository" yerine gerçek URL yazıldı. §8'e eklenen "arşiv `results/cells/` içeriyor, analiz koşusuz reprodüklenebilir" iddiası `git ls-tree v1.0.0` ile doğrulandı (114 cell + 114 meta gerçekten tag'de). · Engellediği görev: 14 (çözüldü) · Ciddiyet: orta
- [x] **[2026-07-26]** `ARXIV.md` ilk yazımında endorsement politikası **bayat kaynaktan** yazılmıştı: statik yardım sayfası (info.arxiv.org/help/endorsement) "kurumsal e-posta + ortak-yazarlık sahiplenme → otomatik onay" izlenimi veriyor, ben de öyle yazdım. Kullanıcı e-posta adresini sorunca yeniden arandı ve arXiv'in **21 Ocak 2026 tarihli blog duyurusu** bulundu: otomatik endorsement artık akademik e-posta **ve** önceden arXiv yazarlığının **ikisini birden** istiyor (Aralık 2025'te Matematik'te başlayıp genelleştirilmiş; gerekçe "bilimsel olmayan gönderimlerde sürdürülemez artış"). Kullanıcının ikisi de yok → **tek yol kişisel endorsement.** E-posta adresi seçimi (gmail vs. `mert@durukan.dev`) endorsement'ı hiç değiştirmiyor; ikisi de akademik kurum sayılmıyor. `ARXIV.md` §2 düzeltildi. **Ders:** PROTOKOL Kural 1'in tazelik maddesi statik dokümantasyon sayfaları için de geçerli — bir kurumun yardım sayfası, o kurumun politika duyurusundan bayat olabilir. Bu çalışmanın konu seçimi de tam aynı hatadan kurtarılmıştı (bkz. GECMIS.md "Neden bu konu"). · Engellediği görev: yok (çözüldü) · Ciddiyet: orta
- [x] **[2026-07-24]** `mxfp8` group size kısıtı doğrulanmadı (mxfp4'ün 32 olduğu kanıtlandı, mxfp8 varsayıldı). Ön-kayıtta mxfp8 koşulu yok, o yüzden şimdilik engel değil. **Kapatıldı (2026-07-26, kapsam dışı):** `mxfp8` ön-kayıtlı 14 koşulun hiçbirinde yer almadı, ölçüm ızgarası tamamlandı (114/114 hücre) ve `PREREG.md` donmuş durumda — bu varsayım hiçbir sonuca girmedi ve girmesi de mümkün değil. Doğrulanmadan kapatılıyor, çünkü doğrulanacak bir iddia kalmadı; ileride bir v2 çalışması mxfp8'i eklerse **o zaman** kanıtlanması gereken bir ön koşul olur (o çalışmanın kendi fizibilite kapısında, Kural 2). · Engellediği görev: yok · Ciddiyet: düşük
- [x] **[2026-07-24]** `mixed_*` recipe'lerin nasıl çağrıldığı (`quant_predicate` string mi, builder fonksiyonu mu) doğrulanmadı. Görev 5'te kanıtlanmalı. **Çözüldü (2026-07-25, Görev 5 öncesi):** `mlx_lm/convert.py` kaynağı okundu + gerçek `convert()` çağrısıyla doğrulandı — `quant_predicate` doğrudan `condition_tag` string'i (`"mixed_2_6"` vb., `QUANT_RECIPES` ile birebir eşleşiyor), `q_mode="affine"` zorunlu, `q_group_size=None`/`q_bits=None` (recipe kendi per-layer bit/group değerlerini döndürüyor, gerçek group size iç varsayılan 64'e düşüyor — `config.json`'da doğrulandı). Detay: GECMIS.md. · Engellediği görev: yok (çözüldü) · Ciddiyet: orta
- [x] **[2026-07-24]** Llama-3.2-1B ve 3B'nin doğruluğu hiç ölçülmedi (sadece erişilebilirliği doğrulandı). Uygunluk kapısı bunu FAZ 1'de zaten yakalayacak, ama havuzun 4'ten aza düşme riski var. **Çözüldü (2026-07-26, Görev 9):** ölçüldü — `llama3.2-1b` eşiğin altında kaldı (arc 0.475/mmlu 0.435, eligible=false), ama **`llama3.2-3b` eşiği geçti** (arc 0.713/mmlu 0.569, eligible=true) ve tam merdiveni işlendi (28 hücre). Havuz 4'ün altına düşmedi: `qwen2.5-1.5b`, `qwen2.5-3b`, `llama3.2-3b` (3 ana model, eligible) + `qwen2.5-0.5b` (floor control). · Engellediği görev: yok (çözüldü) · Ciddiyet: orta
- [x] **[2026-07-24]** MMLU'da seçenek sayısı her zaman 4 mü, doğrulanmadı. `n_options` sabit varsayılmamalı. **Çözüldü (2026-07-25, Görev 4 öncesi):** `cais/mmlu` `all`/test split'i (14042 satır) **çalıştırılarak** tarandı — hepsi tam 4 seçenekli. Ayrıca `allenai/ai2_arc` ARC-Challenge test split'i (1172 satır) de tarandı: seçenek sayısı **{3, 4, 5} arasında değişiyor** — bu da SPEC'teki `Item.options: list[str]` + `n_options` kolonunun sabit varsayılmaması gerektiğini doğruluyor, `load_items`/`run_cell` seçenek sayısını asla 4'e sabitlememeli. Detay: GECMIS.md. · Engellediği görev: yok (çözüldü) · Ciddiyet: orta
- [x] **[2026-07-25]** `tests/test_determinism.py` pytest `assert` içermiyor — top-level script, import edildiğinde (yani her `make test`'te) gerçek bir 1.5B model conversion'ı çalıştırıyor ve sadece print ediyor. `pytest` onu topluyor ama hiçbir test fonksiyonu bulamıyor (0 test), yine de içeriği collection sırasında **çalışıyor** — her `make test` artık ağ/HF-cache'e bağımlı ve yavaş, ve gerçek bir pass/fail sinyali yok. PREREG §4.6.6 sözleşmesi "geçti" olarak commit edilmişti ama pytest formatında değil. **Çözüldü:** `pytest.fixture(scope="module")` + iki gerçek test fonksiyonuna (`test_repeated_inference_is_bit_identical`, `test_injected_noise_is_detected`) çevrildi; sözleşme ve mutasyon kanıtı aynı, artık gerçek pass/fail veriyor. `pytest tests/ -v` → 16/16 passed (bkz. 2 yeni test). · Engellediği görev: yok · Ciddiyet: orta
- [x] **[2026-07-25]** Görev 2 kabul kriteri metni ("hepsi `NotImplementedError` ile FAIL etmeli") ile aynı görevin "Yap" talimatı ("`src/metrics.py`'yi **YAZMA**") çelişiyordu: modül hiç yoksa import `ModuleNotFoundError` ile collection hatası verir, tek tek `NotImplementedError` FAIL'i değil. **Çözüldü:** kullanıcıyla birlikte karar verildi — `src/metrics.py`'ye SPEC §3'ün **imza iskeleti** yazıldı (her gövde yalnızca `raise NotImplementedError`, sıfır mantık). PROTOKOL Kural 3'e bu desen kalıcı ek olarak yazıldı. `pytest tests/test_metrics.py -q` artık 13/13 `NotImplementedError` ile FAIL veriyor — doğrulandı. · Engellediği görev: yok · Ciddiyet: düşük
- [x] **[2026-07-25]** `requirements.txt` içinde `statsmodels` ve `scipy` **yok**, ama SPEC §3 `cal_intercept`'in "`statsmodels` GLM, `offset=` ile" uygulanmasını zorunlu kılıyor. Şu an kurulu değiller (`ModuleNotFoundError`). Görev 2'nin referans hesaplarını (bkz. `tests/test_metrics.py`) numpy-only Newton-Raphson ile bağımsız doğruladım, bu yüzden Görev 2'yi engellemiyor — ama Görev 3 (`src/metrics.py` implementasyonu) bu bağımlılık eklenmeden başlayamaz (SPEC §0 madde 8: sabitlenmemiş bağımlılık eklenemez, `requirements.txt` + `requirements.lock.txt` aynı commit'te güncellenmeli). **Çözüldü:** `statsmodels==0.14.6` + `scipy==1.18.0` eklendi, `requirements.lock.txt` yeniden üretildi (`pip freeze`), GLM `offset=` fit'i fonksiyonel olarak doğrulandı (PROTOKOL Kural 6). · Engellediği görev: 3 · Ciddiyet: orta
- [x] **[2026-07-25]** `tests/test_metrics.py::test_cal_slope_perfect_calibration_is_near_one` içinde gerçek bir hata bulundu: `expected_slope, _ = _fit_reference_cal_slope_intercept(y, conf)` tuple'ı yanlış sırayla açıyordu (fonksiyon `(intercept, slope)` döndürüyor, test `expected_slope`'a intercept'i atıyordu). `metrics.cal_slope`'un doğru olduğu bağımsız doğrulandı (referans `b1` ile ondalık düzeyinde eşleşme: 1.0124437146123444 vs 1.012443714612344). **Çözüldü:** kullanıcı onayıyla tek satır düzeltildi (`_, expected_slope = ...`), bkz. GECMIS.md "Görev 3". · Engellediği görev: yok (çözüldü) · Ciddiyet: orta
- [x] **[2026-07-25]** Görev 6 (`tests/test_no_leakage.py`), SPEC §7'ye göre `runner.py`'den (Görev 7) önce sıralanıyor, ama testin sözleşmeleri ("eligibility.json yokken Faz 3 çalışamaz", "uygunluk yalnızca bf16'dan") `runner.py`'de bir şeye karşı test edilmeyi gerektiriyor — modül hiç yoksa test totolojik/imkansız olurdu. **Çözüldü:** `src/runner.py`'ye SPEC §3'e eklenen iki saf fonksiyonla (`compute_eligibility`, `assert_phase3_allowed`) kısmi implementasyon yazıldı (kuantizasyon parametresi seçmiyor/değiştirmiyor, SPEC §0 madde 1 riski yok); `cell_id`/`run_all` Görev 7'de kalıyor. SPEC §9 Changelog + GECMIS.md "Görev 6". · Engellediği görev: yok (çözüldü) · Ciddiyet: orta
- [x] **[2026-07-25]** `make pilot` çalıştırıldı: bf16 hücresi `status="ok"` (1000 skorlanmış, 206.7s), ama **her iki kuantize hücre de (`affine_b4_g64`, `affine_b2_g64`) `status="failed"`** — hiçbiri dönüşüme bile ulaşmadı (`n_items_scored: 0`, `wall_seconds` <0.4s). Kök neden: `runner.py:79`'daki `_run_and_write_cell`, `out_dir`'i `tempfile.mkdtemp(prefix=...)` ile **önceden oluşturuyor**, ama `mlx_lm.convert()` kaynak kodu (`Path(mlx_path).exists()` kontrolü) hedef dizin **zaten varsa** `ValueError` fırlatıyor ("Cannot save to the path ... as it already exists"). Bu, `runner.py` üzerinden çağrılan **her** non-bf16 hücrede %100 tekrarlanır — kuantizasyon parametresi sorunu değil, saf dizin-yönetimi hatası. Görev 5'in fonksiyonel doğrulaması bunu yakalamadı çünkü `quantize.build()` doğrudan, `tempfile.mkdtemp` kullanmayan sabit bir yol (`/tmp/q_t1`, önceden var olmayan) ile çağrılmıştı — hata yalnızca `runner.py`'nin orkestrasyon yolundan geçince ortaya çıkıyor. Sonuç: **hiçbir kuantize hücre şu ana kadar `runner.py` üzerinden gerçekten hiç çalışmadı**, SPEC §8 kabul kriterinin 2-bit/bf16 sağlık kontrolü hiç görülemedi. `never-raise` sözleşmesi doğru çalıştı (crash olmadı, `status="failed"` + hata metni yazıldı) ama görev tamamlanamaz durumda. **Çözüldü:** `_run_and_write_cell`, `mkdtemp` ile tekil bir isim üretip hemen `os.rmdir` ile siliyor — `mlx_lm.convert()`'e her zaman **yeni ama garantili tekil** bir yol veriliyor. `make pilot` yeniden koşuldu: 3/3 hücre `status="ok"`, `affine_b4_g64` effective_bits=4.501, `affine_b2_g64` effective_bits=2.501; doğruluk+ECE sağlık kontrolü aşağıda (bkz. DURUM.md) — 2-bit belirgin kötü, kuantizasyon gerçekten uygulanıyor. `pytest tests/ -q` → 33 passed, `git diff -- tests/` boş. · Engellediği görev: 8 (çözüldü) · Ciddiyet: kritik
- [x] **[2026-07-25]** `Makefile`'daki `pilot:` hedefi `$(PYTHON) -m src.runner --pilot` çağırıyor (Görev 1'de yazıldı), ama SPEC §3'ün `runner.py` sözleşmesi yalnızca `run_all(force: bool=False)`'ı tanımlıyor — ne bir CLI/`__main__` girişi ne de `--pilot` modu var. Ayrıca SPEC §8/YAPILACAKLAR Görev 8 "bir model" diyor ama **hangi model** hiçbir yerde adlandırılmamış. Görev 7 sırasında `run_all` yazılırken keşfedildi. **Çözüldü (aynı oturum):** `src/runner.py`'ye `run_pilot()` (SPEC §8'deki 3 hücreyi — bf16, affine_b4_g64, affine_b2_g64 × arc_challenge — aynı üç-fazlı sırayla, `_run_and_write_cell`/`compute_eligibility`/`assert_phase3_allowed` yeniden kullanılarak koşturuyor) ve `if __name__=="__main__"` + `argparse --pilot` eklendi; `run_all`'ın imzası **değişmedi** (SPEC sözleşmesi korundu). Model seçimi: `qwen2.5-1.5b` (dört ana modelin en hızlısı, keşif fazında %73 doğrulukla uygunluk barının belirgin üzerinde — bilinçli bir mühendislik kararı, PREREG'e dokunmuyor çünkü pilot verisi §0 Pilot İfşası gereği zaten nihai tablolara girmiyor). SPEC §9 Changelog + GECMIS.md "Görev 7". · Engellediği görev: 8 (çözüldü) · Ciddiyet: orta

- [x] **[2026-07-25]** `Makefile`'daki `reproduce:` hedefi `test && runner && analyze` sırasıyla çalışıyor, ama `src/analyze.py` henüz yazılmadı (Görev 10). `caffeinate -i make reproduce` bu yüzden grid tamamlandıktan **sonra** `ModuleNotFoundError`/`No module named src.analyze` ile beklenen bir hatayla bitecek — veri kaybı yok (runner hücre başına diske zaten yazıyor), sadece `make`'in çıkış kodu 0 olmayacak. Görev 9'un kabul kriteri yalnızca grid'in status dağılımını istiyor, analyze'i değil, o yüzden engel değil. **Çözüldü (2026-07-26, Görev 10):** `src/analyze.py` yazıldı ve gerçek veriyle iki kez koşuldu, `make reproduce` artık `python -m src.analyze` adımında da başarıyla biter. · Engellediği görev: yok (çözüldü) · Ciddiyet: düşük
- [x] **[2026-07-26]** Görev 10 sırasında gerçek 114 hücrelik veri üzerinde `python -m src.analyze` ilk koşusunda `numpy` "Mean of empty slice" / "invalid value encountered in scalar divide" uyarıları çıktı. Kök neden bulundu: `metrics.overconfidence_rate`, eşiği (`OVERCONF_THRESHOLD=0.90`) aşan hiçbir item olmadığında `y_correct[above]` boş diziye düşüyor, `np.mean([])` NaN + uyarı üretiyor — `metrics.py`'de bir hata değil, tanımın kendisinde (0/0) bir dejenere durum. Etkilenen 4 hücre gerçek: `qwen2.5-3b`'nin `affine_b2_g64` ve `mixed_2_6` koşullarında (her iki benchmark'ın bazı kombinasyonlarında) hiçbir tahmin %90 güveni aşmıyor (`max(conf_pred)=0.838`) — bu, aşırı sıkıştırmada modelin "aşırı-güvenli" değil tam tersi **belirsiz** hale geldiğini gösteren gerçek bir bulgu, veri hatası değil. **Çözüldü:** `metrics.py`'ye (Kural 4 kanıtlı, dokunulmadı) değil, `analyze.build_table2`'ye `overconfidence_rate_n_qualifying` kolonu eklendi (eşiği aşan item sayısı) — NaN artık açıklamasız görünmüyor. İkinci koşu temiz (hiç uyarı yok), `pytest tests/` → 62 passed. Detay: GECMIS.md "Görev 10". · Engellediği görev: yok (çözüldü) · Ciddiyet: düşük
- [x] **[2026-07-25]** Görev 9 sürerken (37/140 hücre) `eligibility.json`'da `llama3.2-1b` ve `llama3.2-3b` için **birebir aynı** sayılar görüldü (`arc_challenge: 0.222`, `mmlu: 0.238`, ikisi de) — iki farklı boyutlu modelin bire bir aynı doğruluğu vermesi kendi başına şüpheli, kontrol edildi. Kök neden bulundu ve **kanıtlandı**: `src/measure.py:26`'daki `_score_item`, seçenek harfinin logit indeksini `tokenizer.encode(" " + label)[0]` ile alıyor — yani encode çıktısının **ilk** token'ı. Qwen tokenizer'ında bu doğru (`encode(" A") == [362]`, tek token). Ama Llama-3.2 tokenizer'ı her `encode()` çağrısına otomatik `<|begin_of_text|>` (id 128000) **başa ekliyor** (`encode(" A") == [128000, 362]`) — kod dört seçenek için de **aynı sabit BOS token'ının** logit'ini okuyor, dördü de birebir eşit çıkıyor, softmax sonrası hepsi 0.25 (kontrol edildi: `llama3.2-1b__bf16__arc_challenge.parquet`'te `pred_idx` 1000/1000 satırda sabit `0`, `conf_pred` std=0.0). **Sonuç: Llama-3.2-1B/3B için ölçülen "doğruluk" model kalitesini yansıtmıyor — sabit "her zaman A seç" davranışının seçenek dağılımına denk gelen bir artefakt.** İki bf16 hücresi (`llama3.2-1b`/`llama3.2-3b` × `arc_challenge`/`mmlu`, 4 hücre) ve bunlardan türetilen `eligibility.json` girdileri geçersiz. **Qwen ailesi etkilenmiyor** (tokenizer'ları `encode()`'da BOS eklemiyor, `eligibility.json`'daki Qwen sayıları zaten çeşitli/anlamlı — pilot sonuçlarıyla tutarlı). Arka planda süren Görev 9 koşusu şu an yalnızca (uygun bulunan) Qwen kuantize hücrelerini işliyor, bu yüzden **daha fazla hesaplama israfı sürmüyor** — ama Llama'nın gerçek uygunluğu hâlâ bilinmiyor (düzeltilirse eşiği geçebilir, bu da onu tüm ızgaraya sokar). Düzeltme muhtemelen basit (`[0]` yerine `[-1]` — Llama'da son token her zaman etiketin kendisi: `[128000, 362][-1] == 362`; Qwen'de tek elemanlı listede `[-1] == [0]`, geriye dönük kırmıyor) ama `measure.py` SPEC §3 sözleşmesi altında kritik/korumalı kod, kullanıcı onayı olmadan değiştirilmedi. **Çözüldü (test-önce, PROTOKOL Kural 3+4):** `tests/test_measure.py` yazıldı (gerçek model yüklemeden, BOS-ekleyen/eklemeyen sahte tokenizer'larla), kullanıcıya gösterildi, onay alındı. `src/measure.py`'ye saf `_option_token_ids(tokenizer, labels)` yardımcı fonksiyonu çıkarıldı (`encode(...)[-1]` — etiketin kendi token'ı BOS var/yok fark etmeksizin her zaman son eleman). Mutasyon kanıtı: `[-1]` → `[0]`'a kasten geri döndürüldü, BOS testi doğru sebeple FAIL etti (`128000 not in [128000,128000,128000,128000]`), sonra dosya orijinaline döndürüldü. Gerçek Llama-3.2-1B tokenizer'ıyla fonksiyonel doğrulandı: `_option_token_ids(tok, ['A','B','C','D']) == [362, 426, 356, 423]` (dördü farklı, önceki `[128000]*4` yerine). `pytest tests/` → 35 passed. **Henüz uygulanmadı:** `results/`'taki 4 llama bf16 hücresi ve `eligibility.json` hâlâ eski (bozuk) veriyle — arka planda süren Görev 9 koşusuyla aynı anda ikinci bir model yükleyen süreç başlatmak riskli (paylaşımlı 24GB birleşik bellek), o yüzden düzeltilmiş `measure.py` ile llama hücrelerinin yeniden üretilmesi **Görev 9'un mevcut arka plan koşusu tamamen bitene kadar ertelendi** (plan: 4 dosya silinir, `python -m src.runner` force=False ile tekrar çağrılır — Qwen hücreleri zaten cache'te olduğu için atlanır, yalnızca llama Faz 1 yeniden koşar, Faz 2 doğru `eligibility.json`'ı yeniden yazar, Faz 3 llama uygun çıkarsa onu da işler). **Kapatıldı (2026-07-26):** ilk koşu bitti (88/88 ok), 4 eski llama bf16 dosyası silindi, `python -m src.runner` tekrar çalıştırıldı — Qwen'in 84 hücresi cache'ten atlandı, llama'nın 4 bf16 hücresi düzeltilmiş kodla yeniden ölçüldü. Nihai sayılar: `llama3.2-1b` arc=0.475/mmlu=0.435 (hâlâ eligible=false), `llama3.2-3b` arc=0.713/mmlu=0.569 (**eligible=true**, önceki bozuk ölçümde bu model tamamen elenmiş olacaktı). Faz 3 llama3.2-3b'nin tam merdivenini işledi (28 hücre). Nihai grid: **114/114 hücre `status="ok"`, hiç `failed` yok.** · Engellediği görev: 9 (çözüldü) · Ciddiyet: kritik

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
- [x] **2026-07-26** Görev 9 — `caffeinate -i make reproduce`, kesinti/devam testi geçti; ilk koşu 88/88 hücre `status="ok"` verdi (llama modelleri o an bozuk ölçümden `eligible=false` çıktığı için 140 değil). Koşu sürerken kritik bir açık bulgu bulundu (Llama-3.2 tokenizer'ı `encode()`'a otomatik BOS ekliyor, `measure.py` yanlış token okuyordu — bkz. GECMIS.md), test-önce + mutasyon kanıtıyla kullanıcı onayında düzeltildi (`tests/test_measure.py`, `pytest tests/` → 35 passed). Düzeltmeden sonra 4 eski llama bf16 hücresi silinip yeniden koşuldu: Qwen cache'ten atlandı, llama yeniden ölçüldü. **Nihai: 114/114 hücre `status="ok"`, hiç `failed` yok.** `llama3.2-3b` düzeltilmiş ölçümle eşiği geçti (eligible=true, önceki bozuk veride tamamen elenmiş olacaktı) ve tam merdiveni işlendi; `llama3.2-1b` eşiğin altında kaldı (yalnızca bf16, 2 hücre)
- [x] **2026-07-26** Görev 10 — `src/analyze.py`: SPEC'e altı saf verdict fonksiyonu eklendi (test-önce + iki turda kullanıcı onayı + mutasyon kanıtı), `tests/test_analyze.py` 27 test, `pytest tests/` → 62 passed. Gerçek 114 hücrelik veride iki kez koşuldu; ilk koşu `overconfidence_rate`'in bazı aşırı-sıkıştırma hücrelerinde tanımsız olduğunu buldu (kod hatası değil, `overconfidence_rate_n_qualifying` kolonuyla şeffaflaştırıldı), ikinci koşu temiz. Tablo 1-4 + taban-kontrol tablosu + `verdicts.json` + 3 figür üretildi ve elle incelendi; `qwen2.5-3b`'nin 3-bit hücresinde gerçek, istatistiksel olarak anlamlı bir H1 ihlali (ters-U) bulundu — Görev 13'e not. `matplotlib` eklendi.
- [x] **2026-07-26** Görev 11 — Veri tarama yasağı kontrolü: `src/analyze.py` + çıktılar + `tests/test_analyze.py` elle incelendi, tablolar/figürler PREREG §5/§4.5'e 1:1 örtüşüyor, kayıt-dışı hiçbir test yok, "Exploratory" bölümüne gerek yok — değişiklik yapılmadı.
- [x] **2026-07-26** Görev 12 — `README.md` yazıldı: tek paragraf bulgu (H1 monoton-olmayan gerçek ihlal + H2 model-başına yön tutarsızlığı + H3'ün 6 hücreden 1'inde fark), manşet sayı tablosu (gerçek verilerden doğrulanarak), `make reproduce`, kapsam sınırları. FAZ 6 başladı.
- [x] **2026-07-26** Görev 13 — `paper.md` yazıldı (8 bölüm, 5 tablo + 3 figür gömülü). Related work için 6 gerçek kaynak WebSearch+WebFetch ile tam metin okunarak doğrulandı, her alıntı ≤15 kelime ve programatik olarak sayıldı; yazar isimleri ayrıca doğrulanıp ilk taslaktaki 4 yanlış tahmin düzeltildi. En yakın önceki çalışma (Proskurina vd. 2024) bulundu ve PREREG §1'in "kesişim boş" çerçevesini nüanslandıran açık bulgu olarak kaydedildi. Sonuçlar bölümündeki tüm sayılar CSV/JSON'dan programatik üretildi; kendi ilk taslağımdaki bir elle-sayma hatası (H2 hücre sayıları) `verdicts.json` karşı kontrolüyle yakalanıp düzeltildi.

- [x] **[2026-07-26]** Bağımsız denetim, `tests/` içinde **on koruma testinin başarısız olamadığını** mutasyon testiyle kanıtladı. En ciddisi: `test_bootstrap_ci_forwards_kwargs_to_metric_fn`, SPEC §3'ün kardeş çalışmadan öğrenilmiş diye **özellikle uyardığı** hatayı (`bootstrap_ci`'ın kwargs'ı sessizce düşürmesi) korumak için yazılmıştı — ama eşik olarak `0.9`, yani `config.OVERCONF_THRESHOLD`'un **tam kendisini** seçmişti; kwargs'ı hiç iletmeyen bir implementasyon birebir aynı sayıyı döndürüyor ve test geçiyor. Diğer hayatta kalanlar: `cal_intercept`'in SPEC-zorunlu offset kısıtı, `cal_slope`'un intercept terimi, `mean_conf_by_correctness`'in dönüş sırası (takas edilse H2'nin yarısı ters çevrilirdi, hiçbir test fark etmezdi — `analyze._mean_conf_incorrect` 0. elemanı alıyor), `overconfidence_rate`'in `>` vs `>=` sınırı, `bootstrap_ci`'ın seed'i, `_paired_bootstrap_delta`'nın nokta tahmini (test sabit +0.1 kaydırma kullandığı için bootstrap ortalaması tanımı gereği tam-örneklem farkına eşitti), `ece`'nin boş girdisi, `_intervals_overlap`'in NaN davranışı, ve H1'in komşu-çift kuralı. Ayrıca `tests/test_metrics.py`'de **ölü kod** bulundu: `_fit_reference_cal_intercept_offset` — offset kısıtını sınamak için yazılmış referans — hiç çağrılmıyordu. **Çözüldü:** 7 yeni gerçek test yazıldı + 2 boş test ayırt edici hale getirildi (Kural 9: delege edilmedi), sonra 10 mutant bir pytest eklentisiyle **bellekte** kurulup (hiçbir repo dosyası değiştirilmeden) her birinin **kendi hedef testiyle ve doğru sebeple** öldüğü gösterildi. `pytest tests/` → 75 passed. **Kendi mutasyon düzeneğim de bir kez bozuk çıktı** ve on mutantın onunu da "hayatta kaldı" gösterdi (eklentinin hook parametresi `config` yerine `config_` adlandırıldığı için pytest eklentiyi hiç yüklemiyordu); düzenek artık her koşumda mutantın gerçekten kurulduğunu doğruluyor. · Engellediği görev: yok (çözüldü) · Ciddiyet: **kritik**

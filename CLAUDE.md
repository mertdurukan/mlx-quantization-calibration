# CLAUDE.md — Giriş Noktası

> Bu dosyayı her oturumda **ilk** okursun. Ne yapacağını burası söyler.

---

## 🔁 OTURUM PROTOKOLÜ — "devam et" dendiğinde

Kullanıcı **"devam et"** derse, tam olarak şu sırayı uygula. Sormadan başla.

### Adım 1 — Durumu oku (3 dosya, sırayla)
1. `DURUM.md` — neredeyiz, son oturumda ne oldu
2. `YAPILACAKLAR.md` — **işaretlenmemiş ilk görev** senin görevin
3. `PROTOKOL.md` — hangi kurallar geçerli (11 kural, hepsi zorunlu)

### Adım 2 — Görevi doğrula
- Görevin ön koşulları sağlanmış mı? (`YAPILACAKLAR.md`'de her görevin "Ön koşul" satırı var)
- Sağlanmamışsa: o görevi atla, ön koşulu bir görev olarak `AÇIK BULGULAR`'a yaz, kullanıcıya söyle.

### Adım 3 — Görevi yap
- `SPEC.md`'deki sözleşmeye uy. `PREREG.md`'ye **dokunma** (donmuş).
- Test yazılacak bir görevse: **testi yaz, DUR, kullanıcı onaylasın**, sonra implementasyon
  (PROTOKOL Kural 3).
- Kritik koruma testiyse: **mutasyonla kanıtla** (PROTOKOL Kural 4).

### Adım 4 — Kanıt sun
Her görev bitiminde, `YAPILACAKLAR.md`'deki **kabul kriteri komutunu çalıştır** ve **gerçek
çıktıyı** kullanıcıya göster. "Yaptım" demek kanıt değildir (PROTOKOL Kural 5).

### Adım 5 — Kayıtları güncelle (ATLANMAZ)
1. `YAPILACAKLAR.md` → görevi `[x]` işaretle, tamamlanma tarihini yaz
2. `DURUM.md` → "Şu an" ve "Son oturumda ne oldu" bölümlerini güncelle
3. Bir karar verdiysen veya bir hata yakaladıysan → `GECMIS.md`'ye ekle
4. Ön-kayıttan saptıysan → `DEVIATIONS.md`'ye ekle (append-only)
5. Commit et, mesajda ne yapıldığını yaz

### Adım 6 — Bir sonraki görevi söyle
Kullanıcıya tek cümleyle: sırada ne var.

---

## ⚠️ AÇIK BULGU MEKANİZMASI (bu projenin kendini onarma yolu)

Çalışırken bir eksik, belirsizlik, tutarsızlık veya risk **keşfedersen**:

1. **HEMEN** `YAPILACAKLAR.md` → `## AÇIK BULGULAR` bölümüne ekle. Şu formatta:
   ```
   - [ ] **[TARİH]** <bulgu tek cümle> · Engellediği görev: <no veya "yok"> · Ciddiyet: kritik/orta/düşük
   ```
2. **Sonra** devam et.
3. **Sessizce geçme.** "Küçük bir şey, sonra bakarız" diye düşündüğün an, o düşünceyi yaz.

**Bu mekanizma olmadan proje ölür.** Kardeş ML çalışmasında yakalanan yedi hatanın hepsi,
"küçük bir tuhaflık" diye başladı.

Kritik ciddiyetteki bir bulgu varsa, mevcut görevi **durdur** ve kullanıcıya sor.

---

## Proje kimliği

**Ne:** MLX kuantizasyon merdiveninin (bf16 → 8/6/5/4/3/2-bit → mxfp4 → mixed) bir LLM'in
**güven kalibrasyonunu** nasıl bozduğunu ölçen ön-kayıtlı çalışma.

**Neden:** İki literatür var ve birbirine değmiyor. MLX/kuantizasyon tarafı throughput ve
doğruluk ölçüyor, kalibrasyona hiç bakmıyor. LLM-kalibrasyon tarafı kuantizasyona hiç bakmıyor.
Kesişim boş (Temmuz 2026'da doğrulandı).

**Tez:** Sıkıştırılmış model doğru cevabı hâlâ veriyor olabilir — ama **ne kadar emin olduğunu
biliyor mu?** Yoksa gizlice aşırı-güvenli mi hale geliyor?

**Kardeş çalışma:** `~/github-projects/imbalance-calibration` (tamamlandı, yayında). Aynı imza:
*"Herkes doğruluğa bakıyor; ben modelin ne kadar dürüst olduğuna bakıyorum."* İkisi bir program.

---

## Dosya haritası

| Dosya | Ne | Değiştirilebilir mi |
|---|---|---|
| `CLAUDE.md` | bu dosya, giriş noktası | evet |
| `DURUM.md` | canlı durum — her oturumda güncellenir | **her oturumda** |
| `YAPILACAKLAR.md` | sıralı görevler + açık bulgular | **her oturumda** |
| `PROTOKOL.md` | 11 kural, her biri yaşanmış hatanın karşılığı | nadiren |
| `PREREG.md` | ön-kayıt | ❌ **ASLA** (donmuş) |
| `SPEC.md` | implementasyon sözleşmesi | evet, Changelog ile |
| `DEVIATIONS.md` | sapma defteri | **append-only** |
| `GECMIS.md` | kararlar ve yakalanan hatalar | append-only |
| `results/` | üretilmiş çıktı | ❌ elle değil, kodla |

---

## Ortam

```bash
cd ~/github-projects/mlx-quantization-calibration
source .venv/bin/activate
```

Kurulu: Python 3.14, `mlx-lm 0.31.3`, `mlx 0.32.0`, `datasets 5.0.0`.
Donanım: Apple M4 Pro 24GB. **CUDA yok** — bitsandbytes/GPTQ/AWQ çalışmaz, MLX çalışır.

Kullanıcının `dl` alias'ı var: dizine geçer + venv açar.

---

## Kırmızı çizgiler (SPEC §0'ın özeti)

1. Bir hücre çalışsın diye kuantizasyon parametresi **değiştirme**
2. Hücre **atlama/silme** — başarısızlık veridir
3. Prompt şablonunu **değiştirme** — tek şablon, donmuş
4. `PREREG.md`'yi **düzenleme** — `DEVIATIONS.md`'ye yaz
5. Güven aralığı olmadan sayı **verme**
6. `results/` altını **elle düzenleme**
7. bf16 referansı koşmadan kuantize hücre **başlatma** (uygunluk kapısı sırası)
8. Sabitlenmemiş bağımlılık **ekleme**
9. Git geçmişini **yeniden yazma**
10. Emin değilsen **DUR ve sor**

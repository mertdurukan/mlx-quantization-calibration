# Araştırma Protokolü — Ön-Kayıtlı Çalışma Disiplini

Bu protokol, ML çalışmasında (imbalance-calibration) geliştirildi ve **yedi gerçek hatayı
yakaladı.** Her kural, yaşanmış bir hatanın karşılığıdır. Kurallar sıkıcı görünüyorsa,
bedeli ödenmiş olduğu içindir.

---

## Çekirdek

> **Doğrula, sonra üret. Riski kendin ara. İddiaya değil, koda bak.**

---

## KURAL 1 — Olgu-Kanıt & Tazelik

**Tetik:** Herhangi bir olgusal iddiadan önce (sayı, tarih, literatür durumu, "X şöyledir"
diyen her cümle).

- **Değişebilir bilgi → ara, doğrula.** Bellekten verme.
- **Atıf doğrula.** Her kaynak gerçek ve erişilebilir olmalı. "Muhtemelen vardır" yasaktır.
- **Tazelik kontrolü:** Hızlı gelişen alanlarda 3+ ay eski çerçeve **bayattır.** Yeniden test et.
- **Güven düzeyini işaretle:** kesin / büyük olasılıkla / doğrulanamadı.

**Yaşanmış hata:** Ocak 2026'da "kuantizasyon-kalibrasyon çelişkisi açık" diye bir DL konusu
seçildi. Temmuz'da doğrulandığında **çelişki çözülmüştü.** Ön-kayıt yazılmadan önce kontrol
edildiği için üç hafta kurtarıldı.

---

## KURAL 2 — Ön-Kayıt Fizibilite Kapısı ⚠️ EN ÖNEMLİ

**Ön-kaydı dondurmadan önce, tasarımın uygulanabilir olduğunu KANITLA.**

- Veri kaynağına keşif sorgusu at (havuz gerçekten N örnek veriyor mu?)
- Donanım/kütüphane fizibilite testi çalıştır (araç istediğin ölçümü yapabiliyor mu?)
- Terimleri tanımla (ör. "missing rate" hücre-düzeyi mi satır-düzeyi mi?)

**Yaşanmış hata:** ML çalışmasında ön-kayıt, OpenML'e tek sorgu atılmadan yazıldı. Sonuç:
havuz 10 değil 8 dataset verdi, ve "missing-value rate" tanımsız kaldı. İkisi de DEVIATIONS'a
girdi. 30 dakikalık keşif, iki sapmayı önlerdi.

---

## KURAL 3 — Testler Önce, İki Aşamalı İnceleme

**Kritik metrik/koruma testleri, implementasyondan ÖNCE yazılır ve ONAYLANIR.**

Sıra: test yaz → **bağımsız inceleme** → onay → implementasyon.

Asla: "test yaz ve geçir" (ajan testi implementasyona uydurur; ölçtüğü şey kendi kodudur).

**Yaşanmış hatalar (bu kural sayesinde yakalandı):**
- ECE testi, `y` sıralı üretildiği için doğru kodda bile patlıyordu
- İki sızıntı testi **totolojikti** — `skf`'i `skf` ile karşılaştırıyordu, herhangi bir
  implementasyonda geçerdi
- `bootstrap_ci` kwargs iletmiyordu → `net_benefit` ile çağrılınca çökerdi

---

## KURAL 4 — Mutasyon Zorunluluğu ⚠️

**Kritik bir koruma testi (sızıntı, kalibrasyon, doğruluk garantisi) yazıldığında, o testin
BAŞARISIZ OLABİLDİĞİ kanıtlanmalıdır.**

Yöntem: kasten bozuk bir implementasyon yaz → testi ona karşı koştur → **doğru sebeple**
patladığını göster → bozuk kodu sil.

**Geçen test, çalışan test değildir.** Yalnızca doğru sebeple başarısız olabildiği gösterilen
test güvence sağlar.

**Yaşanmış kanıt:** Sızıntı testi, kasten sızıntılı bir `run_cell`'e karşı
`max sentinel id 10191 >= 10000` ile patladı. Bu olmadan "sızıntı yok" bir iddia olurdu;
bununla **kanıt** oldu.

---

## KURAL 5 — İddiaya Değil Koda Bak

**Ajanın "yaptım" raporu KANIT DEĞİLDİR.** Her görev sonrası, istisnasız:

```bash
git diff HEAD -- tests/    # kritik: BOŞ olmalı
git status && git diff --stat
```

**Yaşanmış hatalar:**
- "Görev 0b tamam" dedi → `requirements.txt` yoktu, Makefile değişmemişti
- "Dosyayı oluşturdum" dedi → **başka klasöre** yazıyordu (hayalet klasör)
- "4 sızıntı testi hazır" dedi → 2'si hiçbir şey ölçmüyordu
- "make verify yeşil" → ortam bozuktu

---

## KURAL 6 — Fonksiyonel Doğrulama, Varlık Değil

**Bir bileşenin çalıştığını en küçük gerçek çağrıyla kanıtla.** Import etmek, çalıştığını
kanıtlamaz.

**Yaşanmış hata:** `import statsmodels` başarılıydı, `import statsmodels.api` çöküyordu
(scipy 1.14'te `_lazywhere` kaldırılmış). `make verify` yeşil verdi, ortam bozuktu.
Düzeltme: verify artık GLM'i `offset=` ile **gerçekten fit ediyor.**

---

## KURAL 7 — Sapma Kaydı Kutsaldır

- **PREREG.md donmuş.** Tek karakter değişmez.
- **DEVIATIONS.md append-only.** Geçmiş girdi düzenlenmez.
- Her sapma şunları içerir: ne değişti, neden, **sonuçlar görülmeden ÖNCE mi SONRA mı**
  karar verildi, hipotezlere etkisi.

"Decided: BEFORE/AFTER seeing results" alanı en önemlisidir.

---

## KURAL 8 — Git Geçmişi Artefakttır

**Geçmişi yeniden yazma.** Ön-kayıt commit'inin değeri, denetlenebilir ve dokunulmamış
olmasından gelir.

**Tek istisna (yaşandı, kayıtlı):** Yanlış attribution (AI ajanı co-author trailer'ı) düzeltmesi
için bir kez rebase yapıldı. İçerik değişmedi (`git diff backup --stat` boş doğrulandı),
tarihler ve ön-kayıt commit'i korundu.

**Önleme:** Yeni projelerde ajan co-author trailer'ını **baştan kapat.** Her commit sonrası:
```bash
git log -1 --format="%b" | grep -i "co-authored" && echo "TEMIZLE" || echo "temiz"
```

---

## KURAL 9 — Ajan Delegasyonu Yasağı (kritik işlerde)

Alt-ajana delege edilen görevlerde denetim bir katman uzaklaşır. **Test yazımı ve koruma
implementasyonu delege edilmez.**

**Yaşanmış hata:** Sızıntı testleri arka plan alt-ajanına delege edildi, ikisi totolojik çıktı.

---

## KURAL 10 — Denetim Ayrımı

**Kodu yazan bağlam, o kodu onaylayamaz.** Aynı model bile olsa, yazarken taşıdığı gerekçeler
onu okurken kör nokta yaratır.

İnceleme, **sıfır bağlamlı yeni bir oturumda** yapılır; incelemeciye yalnızca diff + spec
verilir, yazarın gerekçeleri verilmez.

---

## KURAL 11 — Üretilmiş Çıktı Elle Düzenlenmez

`results/` altındaki tablo/figür dosyaları **koddan üretilir.** Elle düzenlenirse:
1. Bir sonraki `make reproduce` onları geri getirir
2. Repo **reprodüklenemez** hale gelir (kod ile çıktı tutarsız)

Çıktı yanlışsa **üreten kodu** düzelt, sonra yeniden üret.

---

## Definition of Done (her çalışma için)

Bir çalışma "bitti" sayılır ancak:
- (a) tek komutla çalışan repo (`make reproduce`)
- (b) her sayı güven aralıklı
- (c) hipotezi **çürüten** bulgular da raporlanmış
- (d) sınırlar açıkça yazılmış
- (e) başarısız hücreler **raporlanmış, silinmemiş**

# DURUM — Canlı Durum Dosyası

> **Her oturum sonunda güncellenir.** Bu dosya, "neredeyiz" sorusunun tek cevabıdır.
> Son güncelleme: 2026-07-24

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

**FAZ 3'ün başındayız. Tek satır üretim kodu yazılmadı.**

Repoda şu an olanlar: keşif script'leri (fizibilite kanıtı olarak duruyor), `PREREG.md`,
`SPEC.md`, `PROTOKOL.md`, ve bu operasyonel dosyalar. `src/` klasörü **yok**.

Sıradaki görev: `YAPILACAKLAR.md` → **Görev 1** (config + prompt şablonu dondurma).

## Son oturumda ne oldu (2026-07-24)

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
| | | |

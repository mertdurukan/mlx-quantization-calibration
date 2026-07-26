# arXiv gönderimi — hazır paket ve adımlar

> Bu dosya, `paper.md`'nin arXiv'e gönderilebilir hale getirilmesini kaydeder.
> Paket **hazır ve yerel olarak derlendiği doğrulanmış**. Gönderme adımı senin
> hesabınla yapılır — ajan senin adına arXiv'e giriş yapamaz ve yapmamalı.

---

## 1. Paketi üret

```bash
cd ~/github-projects/mlx-quantization-calibration
source .venv/bin/activate
make paper
```

Üretilenler (`build/arxiv/`, git'e girmez — `.gitignore`'da):

| Dosya | Ne |
|---|---|
| `arxiv-submission.tar.gz` | **arXiv'e yüklenecek dosya budur** (`main.tex` + 3 figür, ~490 KB) |
| `main.pdf` | yalnızca senin okuman için (11 sayfa) — arXiv'e **yüklenmez** (aşağıya bakınız) |
| `abstract-for-arxiv-form.txt` | forma yapıştırılacak düz-ASCII abstract (1916 karakter — arXiv sınırı 1920, `build_paper.py` aşılırsa derlemeyi durduruyor) |
| `main.tex` | pandoc'un `paper.md`'den ürettiği LaTeX |

`paper.md` **tek kaynaktır.** `build/arxiv/` altındaki hiçbir şey elle
düzenlenmez — makalede bir değişiklik gerekiyorsa `paper.md` düzenlenir ve
`make paper` yeniden koşulur (PROTOKOL Kural 11'in aynı mantığı).

**Neden PDF yüklenmiyor:** arXiv, TeX kaynağından üretilmiş PDF'i kabul etmiyor —
"arXiv does not accept dvi, PS, or PDF created from TeX/LaTeX source"
(info.arxiv.org/help/submit, 2026-07-26'da doğrulandı). LaTeX kaynağı yüklenir,
arXiv kendisi derler. Bu bir kısıtlama değil avantaj: arXiv derlediği PDF'i sana
**onaylatmadan yayınlamaz**, yani son kontrol noktası sende.

## 2. Ön koşul: endorsement — **kişisel endorsement gerekiyor**

arXiv, bir kullanıcının **ilk** gönderiminde (ve yeni bir kategoriye ilk
gönderiminde) endorsement istiyor. Politika **21 Ocak 2026'da sıkılaştırıldı**
(blog.arxiv.org/2026/01/21/attention-authors-updated-endorsement-policy —
2026-07-26'da okundu; Aralık 2025'te Matematik'te başlayan değişikliğin
genelleştirilmesi). Otomatik endorsement artık **iki koşulun ikisini birden**
istiyor:

1. akademik/araştırma kurumundan bir e-posta adresi, **ve**
2. arXiv'de aynı endorsement alanında yayınlanmış bir makalede önceden yazarlık.

Gerekçe olarak arXiv "bilimsel olmayan gönderimlerde sürdürülemez bir artış"
gösteriyor; kurumsal e-posta **tek başına** artık yeterli kabul edilmiyor.

**Bunun pratik sonucu: e-posta adresi seçimi endorsement'ı değiştirmiyor.**
Ne `mert@durukan.dev` (Google Workspace, kişisel alan adı) ne bir gmail adresi
akademik kurum sayılır; üstelik önceden arXiv yazarlığı da olmadığı için ikinci
koşul zaten sağlanmıyor. Yani **tek yol kişisel endorsement:** cs alanında
yerleşik bir arXiv yazarından, hesabı açtığında arXiv'in sana verdiği endorsement
kodu ile talep etmek. arXiv personeli endorsement veremiyor.

Bunu gönderim gününden **önce** başlat — makaleyi okuyup endorse edebilecek,
cs.LG/cs.CL'de yayın yapmış bir tanıdık (danışman, meslektaş, yazıştığın bir
araştırmacı) gerekiyor. Endorser'ın ya seni tanıması ya da makaleyi incelemesi
bekleniyor.

**Kullanılacak adres:** `mert@durukan.dev` (hesap + yazışma adresi). Endorsement'a
etkisi yok ama kalıcı ve alan adı sana ait — gmail'e göre yazışma için daha
sağlam bir tercih.

## 3. Forma girilecek metadata

| Alan | Değer |
|---|---|
| **Title** | The Calibration Cost of the MLX Quantization Ladder |
| **Authors** | Mert Durukan |
| **Abstract** | `build/arxiv/abstract-for-arxiv-form.txt` içeriğini yapıştır |
| **Primary category** | `cs.LG` (Machine Learning) |
| **Cross-list** | `cs.CL` (Computation and Language) |
| **Comments** | 11 pages, 3 figures, 5 tables. Pre-registered study; design frozen before any calibration number was computed. Code and data: https://github.com/mertdurukan/mlx-quantization-calibration — archived at https://doi.org/10.5281/zenodo.21596523 |
| **License** | CC BY 4.0 önerilir (kod zaten MIT; CC BY, Zenodo kaydıyla ve reprodüklenebilirlik iddiasıyla tutarlı) |
| **DOI alanı** | boş bırak — Zenodo DOI'si kodun/verinin DOI'si, makalenin dergi DOI'si değil |
| **Processor** | pdflatex (arXiv otomatik seçmeli; seçmezse pdflatex seç) |

Primary neden `cs.LG`: çalışma bir ölçüm/kalibrasyon çalışması, dil-modeline
özgü bir NLP görevi katkısı değil. `cs.CL` cross-list, konu LLM olduğu için.

## 4. Adımlar

1. arXiv'de hesap aç / gir → gerekiyorsa endorsement sürecini başlat (§2).
2. **Start New Submission** → *Article* → lisans seç (§3).
3. Dosya yükleme adımında **`arxiv-submission.tar.gz`**'i yükle. Tek dosya,
   içinden `main.tex` + 3 PNG çıkar; arXiv ana dosyayı `main.tex` olarak bulur.
4. arXiv derler. **Ürettiği PDF'i indir ve `build/arxiv/main.pdf` ile karşılaştır.**
   Aynı görünmeliler. Farklıysa gönderme — burada dur.
5. Metadata'yı §3'teki tablodan gir.
6. Önizlemeyi onayla → **Submit**.

## 5. Bilinen artık riskler

- **TeX Live sürüm farkı.** Yerel derleme TeX Live 2026 ile yapıldı; arXiv 2023
  ve 2025 sunuyor (2025 varsayılan). Kullanılan paketlerin hepsi (geometry,
  amsmath, caption, longtable, booktabs, hyperref, xurl, newunicodechar,
  etoolbox, hyphenat, float) 2023'ten çok daha eski ve kararlı, ama **kesin
  değil.** Yakalama noktası §4 adım 4: arXiv'in kendi PDF'ine bakmadan
  onaylamıyorsun.
- **Gönderim geri alınamaz.** Duyurulduktan sonra makale kaldırılamaz, yalnızca
  yeni versiyon (v2) eklenebilir. Duyurudan **önce** "Unsubmit" ile düzeltme
  yapılabilir.
- **Tablolar `\scriptsize`.** Dört sonuç tablosu 7 kolonluk güven aralıklarından
  oluşuyor ve gövde punto'sunda sayfaya sığmıyor. Okunabilir ama küçük; bir
  hakem/okuyucu şikâyet ederse çözüm tabloları landscape'e almak ya da CI'ları
  ayrı bir kolona bölmek olur (`scripts/build_paper.py` → `TABLE_PREAMBLE`).

## 6. Yerel derleme ortamı (bu makinede kurulu)

`make paper` şunlara ihtiyaç duyar; ikisi de bu oturumda kuruldu:

- `pandoc` 3.10.1 → `brew install pandoc`
- TinyTeX (TeX Live 2026) → `~/Library/TinyTeX`, `--no-path` ile kuruldu
  (sudo istemez), `scripts/build_paper.py` binary'yi tam yolla bulur.
  Ek yüklenen paketler: `caption footnotehyper parskip selnolig soul subfig
  ulem xurl newunicodechar hyphenat`.
- `poppler` (PDF'i denetlemek için `pdftotext`) → `brew install poppler`

Bunlar araştırma bağımlılığı değil, yayın araçları — bu yüzden
`requirements.txt`'e **eklenmediler** (SPEC §0 madde 8 sabitlenmiş *Python*
bağımlılıkları hakkında; `make reproduce` bunların hiçbirine ihtiyaç duymaz).

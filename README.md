# Kitap/Paper/Kod RAG Sistemi

Ders kitaplari, akademik paperlar ve GitHub kod repolarindan olusan bir
bilgi tabanini sorgulanabilir hale getiren, edge cihazda (Jetson Orin Nano
Super) servis edilecek bir RAG (Retrieval-Augmented Generation) projesi.

**Bu dosya hem GitHub reposunda hem de yerel yedek klasorunde bulunuyor -
baska bir bilgisayarda devam etmek icin gereken her sey burada.**

---

## Su anki durum (ozet)

| Bilesen | Durum |
|---|---|
| Kitap/paper indeksi | **7410 chunk**, 15 kaynaktan (Qdrant koleksiyonu: `kitaplar`) |
| Kod repo indeksi | **16105 chunk**, 4 repodan (Qdrant koleksiyonu: `kod_repolari`) |
| Retrieval kalitesi | Recall@5 **%81.4**, Recall@10 **%84.7**, MRR **0.662** (reranker ile) |
| Uctan uca RAG | Calisiyor (`ask.py`) - kaynak alintili, halusinasyon korumali |
| Jetson deploy | Yapilmadi (Faz 6, sirada) |
| Fine-tuning | Yapilmadi (Faz 7, opsiyonel) |
| GitHub push | Bkz. asagida |

---

## Mimari ozet

```
PDF (kitap/paper)                      GitHub repo (kod)
      |                                       |
      v                                       v
[convert_pdfs.py]                    [code_chunking.py]
 pymupdf4llm ile markdown             satir-penceresi (kod) veya
 (baslik hiyerarsisi korunur)         baslik-farkinda (.md dokuman) chunking
      |                                       |
      v                                       |
[chunking.py]                                 |
 baslik-farkinda parcalama                    |
 ~300-450 token'lik chunk'lar                 |
      |                                       |
      v                                       v
[build_index.py]                     [build_code_index.py]
 BAAI/bge-m3 embedding                BAAI/bge-m3 embedding
      |                                       |
      v                                       v
Qdrant koleksiyon: "kitaplar"        Qdrant koleksiyon: "kod_repolari"
 (7410 nokta)                         (16105 nokta)  <- AYRI tutuluyor,
      |                                                  kod ve duz metin
      v                                                  farkli semantik
[query.py] retrieve()
 1) bge-m3 ile top-30 aday bul
 2) bge-reranker-v2-m3 ile yeniden sirala, top-5 don
      |
      v
[ask.py]
 top-5 chunk + soru -> Ollama (qwen3:4b-instruct) -> kaynak gosteren cevap
```

## Neden bu secimler

- **Baslik-farkinda chunking**: Naive "her N karakterde kes" yontemi uzun
  kitaplarda basarisiz oluyor cunku ayni kavram onlarca yerde gecebiliyor.
  Her chunk'a ait oldugu baslik gomulur, hem embedding kalitesini artirir
  hem de "kaynak: X kitabi" seklinde alinti yapmayi kolaylastirir.
- **BAAI/bge-m3 embedding**: Coklu dil (Turkce dahil) destekli, 8GB VRAM'e
  rahat sigan bir model. Hem kitap hem kod icin ayni model kullaniliyor
  (basitlik icin - kod-ozel embedding modeli denenmedi).
- **Qdrant yerel mod**: Bu olcekte (~23k chunk) ayri bir sunucu/Docker
  gerektirmiyor; `QdrantClient(path=...)` ayni islevi dosya tabaninda
  saglar. **ONEMLI GOTCHA**: bu modda `client.delete_collection()` diskteki
  veriyi guvenilir sekilde silmiyor - koleksiyonu yeniden olustururken
  MUTLAKA once diskteki `qdrant_local/collection/<isim>` klasorunu fiziksel
  olarak silmek gerekiyor (build_index.py ve build_code_index.py bunu
  otomatik yapiyor, ama elle Qdrant ile oynarken bunu unutma - aksi halde
  eski+yeni veri birlesip cift sayima yol aciyor, bir kez basimiza geldi).
- **Kitap ve kod AYRI koleksiyonlarda**: Kod sorgusu ("bu fonksiyon ne
  yapiyor") ile kitap sorgusu ("bu kavram nedir") farkli retrieval
  semantigine sahip - karistirmak kaliteyi dusurur.
- **Reranker (bge-reranker-v2-m3)**: Embedding-only retrieval Recall@5 %64.4
  iken, iki asamali (embedding top-30 -> rerank top-5) yaklasim %81.4'e
  cikardi. Bedeli: embedding'e gore cok daha yavas, bu yuzden once genis
  bir havuzu daraltip sonra rerank ediyoruz.
- **qwen3:4b-instruct (thinking DEGIL)**: `qwen3:4b` (varsayilan, "thinking"
  reasoning varyanti) basit bir soru icin bile 1000-2500 token'lik gizli
  dusunme zinciri uretiyor - 8GB laptop GPU'da 45-60 saniye/istek. Ayni
  model ailesinin `-instruct` varyanti thinking yapmiyor, ~4 saniyede
  cevap veriyor. **Bu ayrimi kaybetme** - Ollama'da model secerken
  mutlaka `-instruct` etiketli olanini kullan.
- **Jetson'da sadece inference**: Egitim/fine-tuning Kaggle'da (ucretsiz
  T4x2 GPU, haftada 30 saat kota) yapiliyor; Jetson yalnizca serving icin
  kullaniliyor (8GB paylasimli bellek fine-tuning icin yetersiz).

## Donanim rolleri

| Cihaz | Rol |
|---|---|
| RTX 4060 (laptop) | Gelistirme: PDF isleme, chunking, embedding, RAG debug |
| Kaggle (T4x2, ucretsiz) | Agir/uzun suren isler: QLoRA, embedding fine-tuning |
| Jetson Orin Nano Super | Uretim: quantize LLM + embedding modeli serving |

---

## Baska bir bilgisayarda devam etmek icin kurulum

Bu bolum, projeyi sifirdan baska bir Windows makinede ayaga kaldirmak icin.

### 1. Python ve sanal ortam

```powershell
winget install Python.Python.3.11
mkdir locproject
cd locproject
python -m venv rag-env
.\rag-env\Scripts\Activate.ps1
```

Script calismiyorsa: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

### 2. CUDA destekli PyTorch (torch >= 2.6 SART - guvenlik geregi eski
surumlerde `transformers` kutuphanesi calismiyor)

```powershell
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Dogrulama: `python -c "import torch; print(torch.cuda.is_available())"` -> `True` olmali.

### 3. Diger kutuphaneler

```powershell
pip install -r requirements.txt
```

### 4. Ollama kur ve modeli indir

```powershell
winget install Ollama.Ollama
ollama pull qwen3:4b-instruct
```

**DIKKAT**: `qwen3:4b` DEGIL, `qwen3:4b-instruct` indir (yukaridaki not'a bak).

### 5. Veriyi yerlestir

Bu yedek klasorunde zaten `data/pdfs/`, `data/markdown/`, `data/repos/`,
`qdrant_local/` hazir geliyor - hicbir sey yeniden calistirmana gerek yok,
dogrudan sorgulamaya baslayabilirsin (asagidaki "Kullanim" bolumu).

Eger sifirdan (sadece kod ile, GitHub'dan klonlayarak) baslıyorsan:
- Kendi PDF'lerini `data/pdfs/` klasorune koy
- `data/repos.txt`'deki repolari `git clone --depth 1 <url> data/repos/<ad>` ile klonla
- Sirasiyla calistir: `convert_pdfs.py` -> `build_index.py` -> `build_code_index.py`

---

## Kullanim

```powershell
.\rag-env\Scripts\Activate.ps1

# Kitap/paper sorgusu (kaynak alintili LLM cevabi):
python ask.py "op-amp virtual short circuit kavrami nedir"

# Sadece retrieval (LLM'siz, hizli, hangi chunk'lar bulundu gorme):
python query.py "sorunu buraya yaz"

# Kullanilan baglami da gormek icin:
python ask.py --show-context "sorunu buraya yaz"

# Yeni PDF eklediginde (data/pdfs/ klasorune koyduktan sonra):
python convert_pdfs.py
python build_index.py

# Yeni repo eklediginde (data/repos/<ad> klasorune klonladiktan sonra):
python build_code_index.py

# Kod repolarini sorgulamak icin (kitaplardan AYRI koleksiyon, LLM'siz retrieval):
python query_code.py "NSGA-II crossover implementation"

# Retrieval kalitesini olcmek icin (eval seti yoksa once uret):
python generate_eval_set.py --n 60
python evaluate_retrieval.py --rerank
```

---

## Yol haritasi (roadmap)

- [x] **Faz 0 — Ortam kurulumu**: Python, CUDA/torch, gerekli kutuphaneler
- [x] **Faz 1 — Ingest pipeline**: PDF -> markdown -> baslik-farkinda chunking
- [x] **Faz 2 — Indeksleme**: 15 kitap/paper kaynagi (7410 chunk) Qdrant'a
      yazildi. 1 kaynak (`1-s2.0-S0045794901000396-main.pdf`) font-encoding
      bozuklugu nedeniyle (%94.9 gecersiz karakter) `data/excluded/`'a
      tasindi, indekse dahil edilmedi.
- [x] **Faz 3 — Eval seti**: `generate_eval_set.py` ile 60 soru-chunk cifti
      uretildi (1'i anormal/runaway uretim oldugu icin elendi, 59 kaldi -
      script artik 400 karakterden uzun sorulari otomatik eliyor).
      Baseline: Recall@5 %64.4, Recall@10 %71.2, MRR 0.473.
- [x] **Faz 4 — Retrieval iyilestirme**: `reranker.py` (bge-reranker-v2-m3)
      eklendi - iki asamali retrieval (embedding top-30 -> rerank top-5).
      Sonuc: Recall@5 %64.4 -> **%81.4**, Recall@10 %71.2 -> **%84.7**,
      MRR 0.473 -> **0.662**. `query.py`/`ask.py` varsayilan olarak
      reranker kullanir. Kalan kayiplar cogunlukla kitaplardaki "8.66
      numarali soru" tarzi alistirma referanslari.
- [x] **Faz 5 — Generation**: `ask.py` ile uctan uca RAG calisiyor.
      Model: `qwen3:4b-instruct` (Ollama). Iki davranis dogrulandi:
      (1) kaynak alintili, baglama sadik cevap (op-amp virtual short testi)
      (2) baglamda olmayan soruda uydurmuyor, "bu bilgi kaynaklarimda yok"
      diyor (kuantum bilgisayar testi, kitaplarda olmayan bir konu).
      **GitHub repolari da eklendi** (bu fazin daha once eksik kismiydi):
      4 repo (`stm32f429`, `nsganetv2`, `strawberryfields`, `d2l-en`),
      icerik-hash tabanli tekrar-eleme ile (STM32'de 791 duplike dosya
      elendi - ayni HAL/CMSIS surucu dosyalari onlarca ornek projede
      tekrarlaniyordu), AYRI bir Qdrant koleksiyonunda (`kod_repolari`,
      16105 chunk).
- [ ] **Faz 6 — Jetson deploy**: Qdrant snapshot (`qdrant_local/` klasoru,
      her iki koleksiyon dahil) + GGUF model Jetson'a tasima, llama.cpp
      CUDA backend, latency olcumu. **Sirada olan adim bu.**
- [ ] **Faz 7 — Fine-tuning (opsiyonel)**: Once embedding fine-tune
      (Faz 3'teki soru-chunk ciftleriyle, Kaggle'da - ucretsiz T4x2, 30
      saat/hafta kota), sonra istenirse LLM QLoRA (davranis/format icin,
      bilgi icin degil). Laptop yerine Kaggle'da yapilmasinin sebebi:
      QLoRA saatlerce %100 GPU yuku demek, laptop icin sagliksiz/yavas.

---

## Test sonuclari ve bilinen sinirlama (kod retrieval)

Uctan uca dogrulama testleri (kitap + kod koleksiyonlari):

| Test | Kaynak | Sonuc |
|---|---|---|
| FreeRTOS task oncelik ayari | Kitap koleksiyonu | Basarili - dogru API (`vTaskPrioritySet()`), kaynak gosterdi |
| NSGA-II non-dominated sorting | Kitap koleksiyonu (MOEA-D paper) | Basarili - Pareto cephesi, crowding distance dogru anlatildi |
| Attention mechanism / Transformers | Kod koleksiyonu (d2l-en) | Basarili - skor 0.96-0.98, tam dogru bolumler |
| NSGA-II crossover/mutation implementasyonu | Kod koleksiyonu (nsganetv2) | **Zayif** - skor 0.004-0.024 |

**Bilinen sinirlama**: `nsganetv2` reposu crossover/mutation'i kendi
kodlamiyor, `pymoo` kutuphanesinden hazir fonksiyon cagiriyor
(`get_crossover("int_two_point")`, `get_mutation("int_pm")`). Kodun
kendisi kavrami aciklamiyor, sadece bir kutuphane cagrisi - embedding'in
eslesecek anlamsal zenginligi yok. Bu bir bug degil, satir-penceresi
chunking'in duzyazi-agirlikli icerige (d2l-en gibi) karsi seyrek/kutuphane-
devirmeli ham kod uzerindeki dogal siniri. Ileride iyilestirme fikri:
fonksiyonlara LLM ile kisa aciklama ekleyip embedding'e onu da katmak
("contextual retrieval" yaklasimi) - henuz yapilmadi.

## Bilgi kaynaklari (15 kitap/paper + 4 repo)

**Kitaplar/Paperlar** (`data/pdfs/`):
- Microelectronic Circuits (Sedra Smith, 5th ed.)
- Microelectronic Circuits Analysis and Design (Rashid)
- Electric Circuits (Nilsson, 10th ed.)
- The Definitive Guide to ARM Cortex-M3/M4 Processors (Joseph Yiu)
- MOEA/D vs NSGA-II: Analog/RF Circuit Optimization (Gebze Teknik Uni. paper)
- MOEA/D Survey Part I & II (arXiv, Ke Li)
- A Survey on Learnable Evolutionary Algorithms (arXiv)
- A-NSGA-II paper (Computers & Industrial Engineering, kullanici eklendi)
- Quantum Computing for the Quantum Curious (OAPEN acik erisim)
- Introduction to Classical and Quantum Computing (Thomas Wong)
- Mark Rodwell (UCSB) analog/mixed-signal IC tasarim ders notlari (2 dosya)
- FreeRTOS Reference Manual + Mastering the FreeRTOS Kernel

**GitHub repolari** (`data/repos/`, bkz. `data/repos.txt`):
- [MaJerle/stm32f429](https://github.com/MaJerle/stm32f429) - STM32F429 ornekleri
- [mikelzc1990/nsganetv2](https://github.com/mikelzc1990/nsganetv2) - Neural Architecture Search
- [XanaduAI/strawberryfields](https://github.com/XanaduAI/strawberryfields) - Foton tabanli kuantum hesaplama
- [d2l-ai/d2l-en](https://github.com/d2l-ai/d2l-en) - Dive into Deep Learning kitabi

---

## Dosya yapisi

```
locproject/
  config.py              -> tum yol/model ayarlari (tek yerden yonetilir)
  requirements.txt        -> pip bagimliliklari (torch haric, ayri kurulur)
  data/
    pdfs/                 -> ham PDF'ler
    markdown/              -> PDF'lerden cikarilan metin
    repos/                 -> klonlanmis GitHub repolari
    repos.txt              -> repo URL listesi
    excluded/               -> indekse alinmayan bozuk kaynaklar + sebebi
  chunking.py             -> kitap/paper icin baslik-farkinda chunking
  code_chunking.py         -> kod/repo icin chunking (satir penceresi + dedup)
  convert_pdfs.py          -> PDF -> markdown
  build_index.py           -> kitap indeksi olustur (koleksiyon: kitaplar)
  build_code_index.py       -> kod indeksi olustur (koleksiyon: kod_repolari)
  reranker.py              -> bge-reranker-v2-m3 sarmalayicisi
  query.py                 -> kitap retrieval (CLI + retrieve() fonksiyonu)
  query_code.py             -> kod repolari retrieval (CLI)
  ask.py                   -> uctan uca RAG (kitap retrieval + LLM cevap)
  generate_eval_set.py      -> LLM ile otomatik soru-chunk seti uretimi
  evaluate_retrieval.py     -> Recall@5/10, MRR olcumu
  eval_set.json             -> uretilen eval seti (telif nedeniyle git'te yok)
  qdrant_local/             -> vektor veritabani (iki koleksiyon)
```

---

## Onemli not — telif (sadece GitHub reposu icin gecerli)

`data/pdfs/`, `data/markdown/` ve `eval_set.json` GitHub reposunda
`.gitignore` ile haric tutuluyor - kitaplarin kendisi telifli icerik
tasidigi icin GitHub'a yuklenmiyor, sadece pipeline kodu paylasiliyor.

**Bu yerel yedek klasorunde ise hepsi mevcut** (senin kendi kullanimin
icin, flash disk yedegi) - GitHub'a pushlanan versiyon ile bu yedek
klasor arasindaki fark budur.

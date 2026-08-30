# Kitap/Paper RAG Sistemi

Ders kitaplari, akademik paperlar ve GitHub kod repolarindan olusan bir
bilgi tabanini sorgulanabilir hale getiren, edge cihazda (Jetson Orin Nano
Super) servis edilecek bir RAG (Retrieval-Augmented Generation) projesi.

## Mimari ozet

```
PDF (kitap/paper)
      |
      v
[convert_pdfs.py]  --pymupdf4llm-->  markdown metin (baslik hiyerarsisi korunur)
      |
      v
[chunking.py]  --baslik-farkinda parcalama-->  ~300-450 token'lik chunk'lar
      |
      v
[build_index.py]  --BAAI/bge-m3 embedding-->  Qdrant (yerel/gomulu mod)
      |
      v
[query.py]  --kosinus benzerligi-->  en alakali top-k chunk
      |
      v
(sonraki asama) LLM (Ollama, Qwen3) --> kaynak gosteren, baglama sadik cevap
```

## Neden bu secimler

- **Baslik-farkinda chunking**: Naive "her N karakterde kes" yontemi uzun
  kitaplarda basarisiz oluyor cunku ayni kavram onlarca yerde gecebiliyor.
  Her chunk'a ait oldugu baslik gomulur, hem embedding kalitesini artirir
  hem de "kaynak: X kitabi, Bolum 5" seklinde alinti yapmayi kolaylastirir.
- **BAAI/bge-m3 embedding**: Coklu dil (Turkce dahil) destekli, dense+sparse
  hybrid search'e uygun, 8GB VRAM'e rahat sigan bir model.
- **Qdrant yerel mod**: Bu olcekte (~15-20k chunk) ayri bir sunucu/Docker
  gerektirmiyor; `QdrantClient(path=...)` ayni islevi dosya tabanli saglar.
- **Jetson'da sadece inference**: Egitim/fine-tuning Kaggle'da (ucretsiz
  T4x2 GPU, haftada 30 saat kota) yapiliyor; Jetson yalnizca serving icin
  kullaniliyor (8GB paylasimli bellek fine-tuning icin yetersiz).

## Donanim rolleri

| Cihaz | Rol |
|---|---|
| RTX 4060 (laptop) | Gelistirme: PDF isleme, chunking, embedding, RAG debug |
| Kaggle (T4x2, ucretsiz) | Agir/uzun suren isler: QLoRA, embedding fine-tuning |
| Jetson Orin Nano Super | Uretim: quantize LLM + embedding modeli serving |

## Kurulum

```bash
python -m venv rag-env
# Windows:
rag-env\Scripts\activate
# CUDA destekli torch:
pip install torch --index-url https://download.pytorch.org/whl/cu121
# Geri kalan kutuphaneler:
pip install -r requirements.txt
```

PDF'lerini `data/pdfs/` klasorune koy (bu klasor .gitignore'da - kendi
kitaplarini repoya eklemene gerek yok, herkes kendi PDF'leriyle calistirir).

## Kullanim

```bash
python convert_pdfs.py    # PDF -> markdown (data/markdown/ altina yazar)
python build_index.py     # chunking + embedding + Qdrant indeksleme
python query.py "sorunu buraya yaz"   # retrieval testi
```

## Yol haritasi (roadmap)

- [x] **Faz 0 — Ortam kurulumu**: Python, CUDA/torch, gerekli kutuphaneler
- [x] **Faz 1 — Ingest pipeline**: PDF -> markdown -> baslik-farkinda chunking
- [ ] **Faz 2 — Indeksleme**: Tum kitaplari embed edip Qdrant'a yaz, elle
      sorgula, chunk boyutunu/overlap'i kalibre et
- [x] **Faz 3 — Eval seti**: `generate_eval_set.py` ile 60 soru-chunk cifti
      uretildi (1'i anormal uretim oldugu icin elendi, 59 kaldi).
      Baseline: Recall@5 %64.4, Recall@10 %71.2, MRR 0.473.
- [x] **Faz 4 — Retrieval iyilestirme**: `reranker.py` (bge-reranker-v2-m3)
      eklendi - iki asamali retrieval (embedding top-30 -> rerank top-5).
      Sonuc: Recall@5 %64.4 -> **%81.4**, Recall@10 %71.2 -> **%84.7**,
      MRR 0.473 -> **0.662**. `query.py` varsayilan olarak reranker kullanir.
      Kalan kayiplar cogunlukla kitaplardaki "8.66 numarali soru" tarzi
      alistirma referanslari - chunk'lama stratejisi icin ayri bir konu.
- [x] **Faz 5 — Generation**: `ask.py` ile uctan uca RAG calisiyor.
      Model: `qwen3:4b-instruct` (Ollama) - qwen3:4b'nin thinking/reasoning
      varyanti DEGIL (bkz. config.py'deki not: thinking varyanti basit bir
      soruda bile 45-60 saniye suruyordu, instruct varyanti ~4 saniye).
      Iki davranis dogrulandi: (1) kaynak alintili, baglama sadik cevap
      (2) baglamda olmayan soruda uydurmuyor, "bu bilgi kaynaklarimda yok"
      diyor. GitHub repolarinin pipeline'a eklenmesi henuz yapilmadi.
- [ ] **Faz 6 — Jetson deploy**: Qdrant snapshot + GGUF model Jetson'a
      tasima, llama.cpp CUDA backend, latency olcumu
- [ ] **Faz 7 — Fine-tuning (opsiyonel)**: Once embedding fine-tune
      (Faz 3'teki soru-chunk ciftleriyle, Kaggle'da), sonra istenirse
      LLM QLoRA (davranis/format icin, bilgi icin degil)

## Onemli not — telif

`data/pdfs/` ve `data/markdown/` klasorleri kasitli olarak `.gitignore`'da.
Kitaplarin kendisi telifli icerik tasidigi icin repoya eklenmemeli/GitHub'a
yuklenmemeli. Sadece pipeline kodu paylasiliyor; herkes kendi PDF'leriyle
calistirir.

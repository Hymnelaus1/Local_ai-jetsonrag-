"""
Proje genelinde kullanilan yol ve model ayarlari.
Tum klasorler proje kokunun (locproject/) altinda toplanmis durumda:

  data/pdfs/       -> ham PDF kaynaklari (kitaplar, paperlar)   [git'e eklenmez]
  data/markdown/   -> PDF'lerden cikarilan metin (ara urun)     [git'e eklenmez]
  qdrant_local/    -> embedding vektorlerinin tutuldugu yerel veritabani [git'e eklenmez]

PDF'ler ve turevleri telifli kitap icerigi tasidigi icin repo disinda tutulur
(bkz. .gitignore). Kod, herkesin kendi PDF'leriyle ayni pipeline'i calistirabilecegi
sekilde tasarlandi.
"""

import os

# Proje kok klasoru (bu dosyanin bulundugu yer)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_SOURCE_DIR = os.path.join(BASE_DIR, "data", "pdfs")
MD_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "markdown")
QDRANT_LOCAL_PATH = os.path.join(BASE_DIR, "qdrant_local")

COLLECTION_NAME = "kitaplar"

# Coklu dil (TR dahil) destegi iyi olan, 8GB VRAM'e rahat sigan embedding modeli.
# Jetson Orin Nano Super uzerinde de ayni model calistirilacak (tutarlilik icin).
EMBED_MODEL = "BAAI/bge-m3"

# Ollama uzerinden calisan uretim (generation) modeli.
# ONEMLI: "qwen3:4b" (thinking/reasoning varyanti) DEGIL, "qwen3:4b-instruct"
# kullaniyoruz. qwen3:4b, basit bir soru icin bile ~1000-2500 token'lik
# gizli "dusunme" zinciri uretiyor (8GB laptop GPU'da 45-60 saniye/istek).
# qwen3:4b-instruct thinking yapmiyor, ayni model ailesi, ~4 saniyede cevap
# veriyor. Acik kaynak, tamamen yerel calisiyor - kota/ucret yok.
GEN_MODEL = "qwen3:4b-instruct"

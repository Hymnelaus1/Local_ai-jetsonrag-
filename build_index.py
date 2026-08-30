"""
ADIM 2: Chunking + Embedding + Vektor indeksleme.

data/markdown/ altindaki tum dosyalari okur, chunking.py ile parcalara ayirir,
BAAI/bge-m3 embedding modeliyle vektorlere donusturur ve Qdrant'in yerel
(sunucu gerektirmeyen, dosya tabanli) modunda saklar.

Neden Qdrant "local mode": Bu projenin olcegi (~15-20k chunk) icin ayri bir
Qdrant sunucusu/Docker konteyneri kurmaya gerek yok; QdrantClient(path=...)
ayni islevi sagliyor ve kurulumu basitlestiriyor. Ileride veri hacmi cok
buyurse (yuz binlerce chunk) gercek bir Qdrant sunucusuna gecmek yeterli -
kod tarafinda degisecek tek sey QdrantClient'in baglanti sekli.

Kullanim:
    python build_index.py

Not: Bu script calistirilinca mevcut koleksiyonu SIFIRLAR (delete + recreate).
Yeni kitap eklediginde tum indeksi yeniden olusturmak icin tekrar calistir.
"""

import glob
import os
import sys
import uuid

# Windows konsolunun cp1252 kod sayfasi bazi Unicode karakterleri yazdiramadigi
# icin stdout'u UTF-8'e zorluyoruz (convert_pdfs.py'deki ile ayni sebep).
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from config import MD_OUTPUT_DIR, QDRANT_LOCAL_PATH, COLLECTION_NAME, EMBED_MODEL
from chunking import chunk_by_headers


def main():
    md_files = glob.glob(os.path.join(MD_OUTPUT_DIR, "*.md"))
    if not md_files:
        print("Hic .md dosyasi yok. Once convert_pdfs.py calistir.")
        return

    # 1) Tum kitaplari chunk'la, hangi kitaptan geldigini metadata olarak tut
    #    (retrieval sonucunda "kaynak: X kitabi" diye gosterebilmek icin).
    all_chunks = []
    for path in md_files:
        book_name = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            md = f.read()
        chunks = chunk_by_headers(md)
        print(f"  {book_name}: {len(chunks)} chunk")
        for c in chunks:
            all_chunks.append({"text": c, "source": book_name})

    print(f"\nToplam {len(all_chunks)} chunk, {len(md_files)} dosyadan\n")

    # 2) Embedding modelini yukle (ilk calistirmada HuggingFace'ten indirilir).
    print(f"Embedding modeli yukleniyor: {EMBED_MODEL} (ilk seferde indirir, ~2GB)...")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [c["text"] for c in all_chunks]
    print("Embedding uretiliyor...")
    # normalize_embeddings=True -> cosine similarity dogrudan dot product ile
    # hesaplanabiliyor, Qdrant tarafinda da Distance.COSINE ile tutarli.
    embeddings = model.encode(texts, batch_size=16, show_progress_bar=True, normalize_embeddings=True)

    # 3) Qdrant'a yaz (yerel/gomulu mod - sunucu/Docker gerekmez).
    print(f"\nQdrant'a (yerel: {QDRANT_LOCAL_PATH}) yaziliyor...")
    client = QdrantClient(path=QDRANT_LOCAL_PATH)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        COLLECTION_NAME,
        vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb.tolist(),
            payload={"text": c["text"], "source": c["source"]},
        )
        for c, emb in zip(all_chunks, embeddings)
    ]
    client.upsert(COLLECTION_NAME, points)

    count = client.count(COLLECTION_NAME).count
    print(f"Tamam. Koleksiyonda {count} nokta var.")


if __name__ == "__main__":
    main()

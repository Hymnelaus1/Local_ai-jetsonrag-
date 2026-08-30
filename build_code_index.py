"""
GitHub repolarini (data/repos/ altina klonlanmis) kod-farkinda sekilde
chunk'lar, embed eder ve AYRI bir Qdrant koleksiyonuna ("kod_repolari") yazar.

Kitap koleksiyonundan ayri tutuluyor (bkz. config.py'deki not) - retrieval
sirasinda hangi koleksiyonun aranacagina soru turune gore karar verilecek.

Once repoyu klonla:
    git clone --depth 1 <url> data/repos/<repo_adi>
(data/repos.txt'de klonlanmasi beklenen repo listesi tutuluyor)

Kullanim:
    python build_code_index.py
"""

import glob
import os
import shutil
import sys
import uuid

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from config import REPOS_DIR, QDRANT_LOCAL_PATH, COLLECTION_NAME_CODE, EMBED_MODEL
from code_chunking import walk_repo


def main():
    repo_dirs = [d for d in glob.glob(os.path.join(REPOS_DIR, "*")) if os.path.isdir(d)]
    if not repo_dirs:
        print(f"'{REPOS_DIR}' altinda klonlanmis repo bulunamadi.")
        print("Once: git clone --depth 1 <url> data/repos/<repo_adi>")
        return

    print(f"{len(repo_dirs)} repo bulundu: {[os.path.basename(d) for d in repo_dirs]}\n")

    all_chunks = []
    for repo_path in repo_dirs:
        repo_name = os.path.basename(repo_path)
        chunks = walk_repo(repo_path, repo_name)
        all_chunks.extend(chunks)

    print(f"\nToplam {len(all_chunks)} kod/dok chunk'i\n")

    print(f"Embedding modeli yukleniyor: {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [c["text"] for c in all_chunks]
    print("Embedding uretiliyor...")
    # Onceki calistirmada batch_size=16 ile CUDA OOM alindi (bazi kod
    # chunk'lari kitap chunk'larindan daha uzun/farkli token dagilimina
    # sahip olabiliyor). Daha kucuk batch, biraz daha yavas ama guvenli.
    torch.cuda.empty_cache()
    embeddings = model.encode(texts, batch_size=4, show_progress_bar=True, normalize_embeddings=True)

    print(f"\nQdrant'a (yerel: {QDRANT_LOCAL_PATH}, koleksiyon: {COLLECTION_NAME_CODE}) yaziliyor...")

    # ONEMLI: bkz. build_index.py'deki ayni notu - delete_collection() yerel
    # modda diskteki veriyi guvenilir silmiyor, fiziksel klasor silme sart.
    collection_dir = os.path.join(QDRANT_LOCAL_PATH, "collection", COLLECTION_NAME_CODE)
    if os.path.isdir(collection_dir):
        shutil.rmtree(collection_dir)

    client = QdrantClient(path=QDRANT_LOCAL_PATH)

    if client.collection_exists(COLLECTION_NAME_CODE):
        client.delete_collection(COLLECTION_NAME_CODE)

    client.create_collection(
        COLLECTION_NAME_CODE,
        vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb.tolist(),
            payload={"text": c["text"], "source": c["source"], "file_path": c["file_path"]},
        )
        for c, emb in zip(all_chunks, embeddings)
    ]
    client.upsert(COLLECTION_NAME_CODE, points)

    count = client.count(COLLECTION_NAME_CODE).count
    print(f"Tamam. '{COLLECTION_NAME_CODE}' koleksiyonunda {count} nokta var.")


if __name__ == "__main__":
    main()

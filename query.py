"""
ADIM 3: Retrieval testi (henuz LLM yok - sadece "en alakali chunk'lari bul").

Bu script, RAG pipeline'inin "R" (retrieval) kismini tek basina test etmeyi
saglar. Soru embedding'e cevrilir, Qdrant'ta kosinus benzerligine gore en
yakin top_k chunk getirilir. LLM entegrasyonu (Ollama ile cevap uretimi)
bir sonraki asama.

Kullanim:
    python query.py "Adam optimizasyon algoritmasi nasil calisir?"
"""

import sys

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from config import QDRANT_LOCAL_PATH, COLLECTION_NAME, EMBED_MODEL


def search(question: str, top_k: int = 5):
    model = SentenceTransformer(EMBED_MODEL)
    client = QdrantClient(path=QDRANT_LOCAL_PATH)

    q_emb = model.encode([question], normalize_embeddings=True)[0]
    hits = client.query_points(COLLECTION_NAME, query=q_emb.tolist(), limit=top_k).points

    print(f"\nSoru: {question}\n")
    for h in hits:
        # score: kosinus benzerligi (1.0'a ne kadar yakinsa o kadar alakali)
        print(f"[{h.score:.3f}] ({h.payload['source']}) {h.payload['text'][:200]}")
        print("---")


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "buraya soru yaz"
    search(q)

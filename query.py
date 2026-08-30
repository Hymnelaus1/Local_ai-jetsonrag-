"""
ADIM 3: Retrieval testi (henuz LLM yok - sadece "en alakali chunk'lari bul").

Faz 4 itibariyle iki asamali retrieval kullanir:
  1) bge-m3 ile genis bir aday havuzu (top-30) bulunur
  2) bge-reranker-v2-m3 ile bu havuz yeniden siralanir, en iyi top-k donulur
Bu yaklasim, evaluate_retrieval.py ile olculdu: Recall@5'i %64.4 -> %81.4'e
cikardigi icin varsayilan davranis oldu. Sadece embedding ile hizli test
icin --no-rerank kullan.

Kullanim:
    python query.py "Adam optimizasyon algoritmasi nasil calisir?"
    python query.py --no-rerank "hizli test"
"""

import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from config import QDRANT_LOCAL_PATH, COLLECTION_NAME, EMBED_MODEL

RERANK_POOL_SIZE = 30


def search(question: str, top_k: int = 5, use_rerank: bool = True):
    model = SentenceTransformer(EMBED_MODEL)
    client = QdrantClient(path=QDRANT_LOCAL_PATH)

    q_emb = model.encode([question], normalize_embeddings=True)[0]

    print(f"\nSoru: {question}\n")

    if use_rerank:
        from reranker import rerank as rerank_fn

        candidates = client.query_points(
            COLLECTION_NAME, query=q_emb.tolist(), limit=RERANK_POOL_SIZE
        ).points
        reranked = rerank_fn(question, candidates, top_k=top_k)
        for c, score in reranked:
            print(f"[{score:.3f}] ({c.payload['source']}) {c.payload['text'][:200]}")
            print("---")
    else:
        hits = client.query_points(COLLECTION_NAME, query=q_emb.tolist(), limit=top_k).points
        for h in hits:
            print(f"[{h.score:.3f}] ({h.payload['source']}) {h.payload['text'][:200]}")
            print("---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*", help="Sorulacak soru")
    parser.add_argument("--no-rerank", action="store_true", help="Reranker kullanma, sadece embedding")
    args = parser.parse_args()

    q = " ".join(args.question) if args.question else "buraya soru yaz"
    search(q, use_rerank=not args.no_rerank)

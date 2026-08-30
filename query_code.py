"""
Kod repolari (kod_repolari koleksiyonu) icin retrieval testi.

query.py'nin kod-koleksiyonu karsiligi - kitaplar yerine GitHub
repolarindan (stm32f429, nsganetv2, strawberryfields, d2l-en) en alakali
chunk'lari bulur. Ayni embedding+rerank mantigini kullanir.

Kullanim:
    python query_code.py "NSGA-II crossover mutation nasil calisir"
"""

import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from config import QDRANT_LOCAL_PATH, COLLECTION_NAME_CODE, EMBED_MODEL

RERANK_POOL_SIZE = 30


def search(question: str, top_k: int = 5, use_rerank: bool = True):
    model = SentenceTransformer(EMBED_MODEL)
    client = QdrantClient(path=QDRANT_LOCAL_PATH)

    q_emb = model.encode([question], normalize_embeddings=True)[0]

    print(f"\nSoru: {question}\n")

    if use_rerank:
        from reranker import rerank as rerank_fn

        candidates = client.query_points(
            COLLECTION_NAME_CODE, query=q_emb.tolist(), limit=RERANK_POOL_SIZE
        ).points
        reranked = rerank_fn(question, candidates, top_k=top_k)
        for c, score in reranked:
            print(f"[{score:.3f}] ({c.payload['source']}) {c.payload['file_path']}")
            print(c.payload['text'][:250])
            print("---")
    else:
        hits = client.query_points(COLLECTION_NAME_CODE, query=q_emb.tolist(), limit=top_k).points
        for h in hits:
            print(f"[{h.score:.3f}] ({h.payload['source']}) {h.payload['file_path']}")
            print(h.payload['text'][:250])
            print("---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*")
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()
    q = " ".join(args.question) if args.question else "buraya soru yaz"
    search(q, use_rerank=not args.no_rerank)

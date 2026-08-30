"""
ADIM 3: Retrieval testi (henuz LLM yok - sadece "en alakali chunk'lari bul").

Faz 4 itibariyle iki asamali retrieval kullanir:
  1) bge-m3 ile genis bir aday havuzu (top-30) bulunur
  2) bge-reranker-v2-m3 ile bu havuz yeniden siralanir, en iyi top-k donulur
Bu yaklasim, evaluate_retrieval.py ile olculdu: Recall@5'i %64.4 -> %81.4'e
cikardigi icin varsayilan davranis oldu. Sadece embedding ile hizli test
icin --no-rerank kullan.

retrieve() fonksiyonu ask.py (Faz 5, LLM cevap uretimi) tarafindan da
kullaniliyor - retrieval mantigi tek yerde tutuluyor.

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

_embed_model = None
_qdrant_client = None


def _get_clients():
    """Embedding modeli ve Qdrant client'ini tembel yukler (tekrar tekrar yuklememek icin)."""
    global _embed_model, _qdrant_client
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path=QDRANT_LOCAL_PATH)
    return _embed_model, _qdrant_client


def retrieve(question: str, top_k: int = 5, use_rerank: bool = True):
    """
    Soru icin en alakali top_k chunk'i doner.
    Donus: [{"text": ..., "source": ..., "score": ...}, ...] formatinda liste.
    """
    model, client = _get_clients()
    q_emb = model.encode([question], normalize_embeddings=True)[0]

    if use_rerank:
        from reranker import rerank as rerank_fn

        candidates = client.query_points(
            COLLECTION_NAME, query=q_emb.tolist(), limit=RERANK_POOL_SIZE
        ).points
        reranked = rerank_fn(question, candidates, top_k=top_k)
        return [
            {"text": c.payload["text"], "source": c.payload["source"], "score": float(score)}
            for c, score in reranked
        ]
    else:
        hits = client.query_points(COLLECTION_NAME, query=q_emb.tolist(), limit=top_k).points
        return [
            {"text": h.payload["text"], "source": h.payload["source"], "score": h.score}
            for h in hits
        ]


def search(question: str, top_k: int = 5, use_rerank: bool = True):
    """CLI icin: retrieve() sonuclarini ekrana yazdirir."""
    results = retrieve(question, top_k=top_k, use_rerank=use_rerank)
    print(f"\nSoru: {question}\n")
    for r in results:
        print(f"[{r['score']:.3f}] ({r['source']}) {r['text'][:200]}")
        print("---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*", help="Sorulacak soru")
    parser.add_argument("--no-rerank", action="store_true", help="Reranker kullanma, sadece embedding")
    args = parser.parse_args()

    q = " ".join(args.question) if args.question else "buraya soru yaz"
    search(q, use_rerank=not args.no_rerank)

"""
ADIM 5 (Faz 3/4): Retrieval kalitesini olcme.

generate_eval_set.py ile uretilen soru setini kullanarak, her soru icin
retrieval'in dogru chunk'i (ground truth) bulup bulamadigini olcer.

Metrikler:
  - Recall@k: Dogru chunk, donen ilk k sonuc icinde mi? (k=5 ve k=10 icin)
  - MRR (Mean Reciprocal Rank): Dogru chunk kacinci sirada geldi (1/rank
    ortalamasi). 1.0'a yakinsa dogru chunk hep ilk sirada demektir.

--rerank bayragiyla calistirilirsa, iki asamali retrieval kullanilir:
  1) bge-m3 ile genis bir havuz (RERANK_POOL_SIZE aday) bulunur
  2) bge-reranker-v2-m3 ile bu havuz yeniden siralanir, en iyi top-k alinir
Boylece reranker'in gercekten katki saglayip saglamadigini AYNI eval
setiyle, sayisal olarak karsilastirabiliyoruz (kor iyilestirme yapmiyoruz).

Kullanim:
    python evaluate_retrieval.py                # sadece embedding (baseline)
    python evaluate_retrieval.py --rerank        # embedding + reranker
"""

import argparse
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from config import QDRANT_LOCAL_PATH, COLLECTION_NAME, EMBED_MODEL

RERANK_POOL_SIZE = 30  # reranker'a verilecek genis aday havuzu buyuklugu


def main(eval_path="eval_set.json", top_k=10, use_rerank=False):
    with open(eval_path, encoding="utf-8") as f:
        eval_set = json.load(f)

    mode = "embedding + reranker" if use_rerank else "sadece embedding (baseline)"
    print(f"{len(eval_set)} soru ile retrieval degerlendiriliyor... [{mode}]\n")

    model = SentenceTransformer(EMBED_MODEL)
    client = QdrantClient(path=QDRANT_LOCAL_PATH)

    if use_rerank:
        from reranker import rerank as rerank_fn

    recall_at_5 = 0
    recall_at_10 = 0
    reciprocal_ranks = []
    misses = []

    for item in eval_set:
        q_emb = model.encode([item["question"]], normalize_embeddings=True)[0]

        if use_rerank:
            # Genis havuzu embedding ile bul, sonra reranker ile daralt/sirala.
            candidates = client.query_points(
                COLLECTION_NAME, query=q_emb.tolist(), limit=RERANK_POOL_SIZE
            ).points
            reranked = rerank_fn(item["question"], candidates, top_k=top_k)
            returned_ids = [str(c.id) for c, _score in reranked]
        else:
            hits = client.query_points(COLLECTION_NAME, query=q_emb.tolist(), limit=top_k).points
            returned_ids = [str(h.id) for h in hits]

        expected_id = item["expected_chunk_id"]

        if expected_id in returned_ids:
            rank = returned_ids.index(expected_id) + 1
            reciprocal_ranks.append(1 / rank)
            if rank <= 5:
                recall_at_5 += 1
            if rank <= 10:
                recall_at_10 += 1
        else:
            reciprocal_ranks.append(0)
            misses.append(item)

    n = len(eval_set)
    print(f"Recall@5:  {recall_at_5}/{n}  ({100 * recall_at_5 / n:.1f}%)")
    print(f"Recall@10: {recall_at_10}/{n}  ({100 * recall_at_10 / n:.1f}%)")
    print(f"MRR:       {sum(reciprocal_ranks) / n:.3f}")

    if misses:
        print(f"\n--- Bulunamayan {len(misses)} soru ---")
        for m in misses[:10]:
            print(f"  ({m['expected_source']}) {m['question']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerank", action="store_true", help="Reranker ile iki asamali retrieval kullan")
    parser.add_argument("--eval-path", default="eval_set.json")
    args = parser.parse_args()
    main(eval_path=args.eval_path, use_rerank=args.rerank)

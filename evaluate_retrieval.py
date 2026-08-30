"""
ADIM 5 (Faz 3): Retrieval kalitesini olcme.

generate_eval_set.py ile uretilen soru setini kullanarak, her soru icin
retrieval'in dogru chunk'i (ground truth) bulup bulamadigini olcer.

Metrikler:
  - Recall@k: Dogru chunk, donen ilk k sonuc icinde mi? (k=5 ve k=10 icin)
  - MRR (Mean Reciprocal Rank): Dogru chunk kacinci sirada geldi (1/rank
    ortalamasi). 1.0'a yakinsa dogru chunk hep ilk sirada demektir.

Bu script, chunk boyutu / embedding modeli / hybrid search gibi degisiklikleri
"kor iyilestirme" yapmadan, sayilarla degerlendirmemizi sagliyor.

Kullanim:
    python evaluate_retrieval.py
"""

import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from config import QDRANT_LOCAL_PATH, COLLECTION_NAME, EMBED_MODEL


def main(eval_path="eval_set.json", top_k=10):
    with open(eval_path, encoding="utf-8") as f:
        eval_set = json.load(f)

    print(f"{len(eval_set)} soru ile retrieval degerlendiriliyor...\n")

    model = SentenceTransformer(EMBED_MODEL)
    client = QdrantClient(path=QDRANT_LOCAL_PATH)

    recall_at_5 = 0
    recall_at_10 = 0
    reciprocal_ranks = []
    misses = []

    for item in eval_set:
        q_emb = model.encode([item["question"]], normalize_embeddings=True)[0]
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
        print(f"\n--- Bulunamayan {len(misses)} soru (chunk boyutu/embedding icin ipucu) ---")
        for m in misses[:10]:
            print(f"  ({m['expected_source']}) {m['question']}")


if __name__ == "__main__":
    main()

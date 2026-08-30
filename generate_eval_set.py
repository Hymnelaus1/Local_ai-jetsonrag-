"""
ADIM 4 (Faz 3): Eval seti uretimi.

Amac: Retrieval kalitesini olceceğimiz bir soru-cevap seti olusturmak.
Elle soru yazmak yerine, indeksteki chunk'lardan rastgele bir orneklem alip
Ollama'daki (Qwen3-4B) modele "bu metne bu bilgiyi arayan biri ne sorar"
diye sorduruyoruz. Boylece hem hangi chunk'in dogru cevap oldugunu (ground
truth) hem de soruyu ayni anda elde ediyoruz.

Bu eval seti iki amaca hizmet eder:
  1. Faz 3: evaluate_retrieval.py ile Recall@k / MRR olcumu
  2. Ileride (Faz 7): embedding fine-tuning icin egitim verisi

Kullanim:
    python generate_eval_set.py --n 60
"""

import argparse
import json
import random
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import ollama
from qdrant_client import QdrantClient

from config import QDRANT_LOCAL_PATH, COLLECTION_NAME, GEN_MODEL

PROMPT_TEMPLATE = """Asagidaki metne dayanarak, bu bilgiyi arayan bir kullanicinin sorabilecegi
1 adet Turkce soru uret. Soru metindeki bilgiyi net sekilde hedeflemeli,
genel/belirsiz olmamali (ornegin "bu bolumde ne anlatiliyor" gibi sorma).
Sadece soruyu yaz, baska hicbir aciklama, on-yazi veya <think> etiketi ekleme.

Metin:
{chunk_text}

Soru:"""


def clean_response(text: str) -> str:
    """Qwen3 'thinking' modeli bazen <think>...</think> bloğu ekliyor, temizle."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip().strip('"').strip()


def generate_question(chunk_text: str) -> str:
    resp = ollama.generate(
        model=GEN_MODEL,
        prompt=PROMPT_TEMPLATE.format(chunk_text=chunk_text[:1500]),
    )
    return clean_response(resp["response"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60, help="Kac soru uretilecek")
    parser.add_argument("--out", default="eval_set.json")
    args = parser.parse_args()

    client = QdrantClient(path=QDRANT_LOCAL_PATH)

    # Tum chunk'lari (id + payload) cek - bu olcekte (~5-6k) sorun degil.
    all_points = []
    offset = None
    while True:
        points, offset = client.scroll(COLLECTION_NAME, limit=500, offset=offset, with_payload=True)
        all_points.extend(points)
        if offset is None:
            break

    print(f"Indekste toplam {len(all_points)} chunk var.")

    sample = random.sample(all_points, min(args.n, len(all_points)))
    print(f"{len(sample)} chunk'tan soru uretiliyor...\n")

    eval_set = []
    for i, point in enumerate(sample, 1):
        chunk_text = point.payload["text"]
        source = point.payload["source"]
        try:
            question = generate_question(chunk_text)
        except Exception as e:
            print(f"  [{i}/{len(sample)}] HATA: {e}")
            continue

        # Bazen model "runaway" bir uretime girip alakasiz/dev bir metin
        # dokebiliyor (gozlemlendi: bir chunk icin 120k+ karakterlik alakasiz
        # bir hikaye uretti). Gercek bir soru birkaç yuz karakteri gecmez,
        # bu esigin uzerini anormal kabul edip atliyoruz.
        if not question or len(question) < 5 or len(question) > 400:
            print(f"  [{i}/{len(sample)}] atlandi (bos/gecersiz/anormal uzun soru, uzunluk={len(question)})")
            continue

        eval_set.append({
            "question": question,
            "expected_chunk_id": str(point.id),
            "expected_source": source,
            "chunk_preview": chunk_text[:150],
        })
        print(f"  [{i}/{len(sample)}] ({source}) {question}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, ensure_ascii=False, indent=2)

    print(f"\n{len(eval_set)} soru-cevap cifti '{args.out}' dosyasina yazildi.")


if __name__ == "__main__":
    main()

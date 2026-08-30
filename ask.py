"""
ADIM 6 (Faz 5): Uctan uca RAG - retrieval + LLM cevap uretimi.

Akis:
  1) query.py.retrieve() ile en alakali top-5 chunk'i bul (embedding + reranker)
  2) Bu chunk'lari, hangi kitaptan geldiklerini belirterek prompt'a goum
  3) Ollama (qwen3:4b-instruct) ile cevap uret

Prompt tasarim kurallari (bilinclil):
  - Model SADECE verilen baglamdaki bilgiyi kullanmali, kendi genel bilgisinden
    "halusinasyon" uretmemeli.
  - Baglamda cevap yoksa bunu acikca soylemeli ("bu bilgi kaynaklarimda yok"),
    uydurmamali.
  - Cevabin sonunda hangi kaynaktan (kitap) yararlandigini belirtmeli - boylece
    kullanici iddiayi orijinal kitaptan dogrulayabilir.

Kullanim:
    python ask.py "op-amp virtual short nedir?"
"""

import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import ollama

from config import GEN_MODEL
from query import retrieve

SYSTEM_PROMPT = """Sen, verilen kitap ve makale pasajlarina dayanarak soru cevaplayan bir asistansin.

Kurallar:
- SADECE asagida verilen baglam pasajlarindaki bilgiyi kullan. Kendi genel
  bilgini veya baglamda olmayan bir seyi kesinlikle ekleme.
- Baglamda sorunun cevabi yoksa, uydurma - acikca "Bu bilgi verdigim
  kaynaklarda yok" de.
- Cevabinin sonunda, hangi kaynak(lar)dan yararlandigini belirt
  (orn: "Kaynak: Sedra Smith, Microelectronic Circuits").
- Turkce ve teknik olarak dogru bir uslup kullan."""


def build_context(chunks: list) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] (Kaynak: {c['source']})\n{c['text']}")
    return "\n\n".join(parts)


def ask(question: str, top_k: int = 5, use_rerank: bool = True) -> str:
    chunks = retrieve(question, top_k=top_k, use_rerank=use_rerank)
    context = build_context(chunks)

    user_prompt = f"""Baglam:
{context}

Soru: {question}

Cevap:"""

    response = ollama.chat(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"], chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*", help="Sorulacak soru")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--show-context", action="store_true", help="Kullanilan chunk'lari da goster")
    args = parser.parse_args()

    question = " ".join(args.question) if args.question else input("Soru: ")

    answer, chunks = ask(question, use_rerank=not args.no_rerank)

    print(f"\nSoru: {question}\n")
    print(f"Cevap:\n{answer}\n")

    if args.show_context:
        print("--- Kullanilan baglam ---")
        for c in chunks:
            print(f"[{c['score']:.3f}] ({c['source']}) {c['text'][:150]}")
            print("---")


if __name__ == "__main__":
    main()

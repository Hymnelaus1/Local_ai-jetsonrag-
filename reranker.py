"""
Reranker modulu (Faz 4).

Neden reranker gerekiyor: Embedding tabanli arama (bge-m3) genis bir aday
havuzunu hizli bulmakta iyi ama ince ayrimda (hangi chunk gercekten en
alakali) zayif kalabiliyor. bge-reranker-v2-m3, soru+chunk ciftini birlikte
(cross-encoder olarak) degerlendirdigi icin cok daha isabetli bir siralama
yapiyor - bedeli, embedding'e gore cok daha yavas olmasi (bu yuzden once
genis bir havuzu (orn. top-30) embedding ile daraltiyoruz, sonra sadece o
30 adayi reranker'a veriyoruz).

Iki asamali akis:
  1) Genis aglama: Qdrant + bge-m3 ile top-N aday bul (hizli, ~genel alakalı)
  2) Daraltma: bge-reranker-v2-m3 ile bu adaylari yeniden sirala (yavas ama
     kesin), en iyi top-k'yi don
"""

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from FlagEmbedding import FlagReranker

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

_reranker = None


def get_reranker():
    """Reranker modelini tembel (lazy) yukler - sadece gercekten kullanildiginda."""
    global _reranker
    if _reranker is None:
        print(f"Reranker modeli yukleniyor: {RERANK_MODEL} (ilk seferde indirir, ~1.1GB)...")
        _reranker = FlagReranker(RERANK_MODEL, use_fp16=True)
    return _reranker


def rerank(question: str, candidates: list, top_k: int = 5):
    """
    candidates: Qdrant'tan donen hit listesi (h.payload['text'], h.id iceren nesneler).
    Donus: candidates'in bir alt kumesi, reranker skoruna gore azalan sirada.
    """
    reranker = get_reranker()
    pairs = [[question, c.payload["text"]] for c in candidates]
    scores = reranker.compute_score(pairs, normalize=True)

    # Tek aday varsa compute_score bir liste degil skaler donebilir - normalize et.
    if not isinstance(scores, list):
        scores = [scores]

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

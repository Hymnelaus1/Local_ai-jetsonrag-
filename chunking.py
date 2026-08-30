"""
Baslik-farkinda (header-aware) metin parcalama (chunking).

Neden basit "her N karakterde bir kes" yerine bu yontem:
Akademik kitaplarda ayni kavram onlarca farkli yerde gecebiliyor (ornegin
"gradyan inisi" 40 farkli sayfada anlatilabilir). Chunk'a hangi basligin
altinda oldugunu (## Bolum 5: Optimizasyon gibi) gommek, embedding modelinin
o parcayi dogru baglamda temsil etmesini saglar ve retrieval kalitesini
belirgin sekilde artirir.

Akis:
1. Metni markdown basliklarindan (#, ##, ###) boler.
2. Her bolumu, basligini koruyarak max_chars karakterlik parcalara ayirir.
3. Her parcanin basina ait oldugu baslik satirini tekrar ekler; boylece
   embedding'e giden metin hem icerigi hem baglami tasir.
"""

import re


def chunk_by_headers(md: str, max_chars: int = 1800) -> list[str]:
    """
    Markdown metnini baslik sinirlarina saygili sekilde parcalara ayirir.

    Args:
        md: pymupdf4llm.to_markdown() ciktisi (veya benzer markdown metni).
        max_chars: Bir chunk'in en fazla karakter uzunlugu. 1800 karakter
            kabaca 300-450 token'a denk gelir; RAG icin makul bir aralik.

    Returns:
        Her biri "## Baslik\\nicerik..." formatinda string listesi.
    """
    # Yeni bir baslik satirindan once metni kes (lookahead ile basligi kaybetme)
    sections = re.split(r'\n(?=#{1,3} )', md)

    chunks = []
    for sec in sections:
        header_match = re.match(r'(#{1,3} .+)\n', sec)
        header = header_match.group(1) if header_match else ""
        body = sec[len(header):] if header_match else sec

        # Uzun bolumleri max_chars'a gore alt parcalara bol, her parcaya
        # ait oldugu basligi yeniden ekle (baglam kaybolmasin diye).
        for i in range(0, len(body), max_chars):
            piece = body[i:i + max_chars].strip()
            if piece:
                chunks.append(f"{header}\n{piece}" if header else piece)

    return chunks

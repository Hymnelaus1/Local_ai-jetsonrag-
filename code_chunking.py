"""
Kod dosyalari icin chunking (Faz 5'in tamamlanmamis parcasi: GitHub repolari).

Kitaplardaki "baslik-farkinda" chunking mantigi kod icin uygun degil - kod
markdown basligi tasimaz. Bunun yerine:

  - .md / .rst dosyalari (README, dokumantasyon): duz metin gibi, mevcut
    chunk_by_headers() (markdown icin) ya da satir-penceresi (rst icin)
    kullanilir.
  - Kod dosyalari (.py, .c, .h, vb.): satir penceresi (line-window) ile
    bolunur; her chunk'in basina "Dosya: <goreli yol>" bilgisi eklenir.
    Bu, kitaplardaki "baslik prefix'i" numarasinin kod karsiligi - embedding
    modeline hangi dosyadan geldigini soyler, boylece retrieval hem icerigi
    hem baglami (dosya adi, konum) degerlendirebilir.

Fonksiyon/sinif sinirlarina gore (AST tabanli) bolme daha hassas olurdu ama
C/Python/RST gibi cok farkli dilller icin ayri ayri parser yazmak bu asamada
gereksiz karmasiklik - satir penceresi cogu durumda yeterli kaliteyi veriyor
ve dil-bagimsiz calisiyor.
"""

import hashlib
import os

from chunking import chunk_by_headers

# Kod olarak islenecek uzantilar. Derlenmis/uretilmis/binary dosyalar,
# gorseller, veri dosyalari (npz, npy) disarida birakiliyor.
CODE_EXTENSIONS = {".py", ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".ipynb"}
DOC_EXTENSIONS = {".md", ".rst", ".txt"}

# Bu klasor adlarini (veya icindeki hicbir seyi) hic gezme - genelde
# vendor/uretilmis kod, bagimlilik, derleme ciktisi barindirirlar.
SKIP_DIR_NAMES = {
    ".git", "__pycache__", "node_modules", "build", "dist", ".venv", "venv",
    "egg-info",
}

LINES_PER_CHUNK = 80
OVERLAP_LINES = 10


def should_skip_dir(dirname: str) -> bool:
    lower = dirname.lower()
    return dirname in SKIP_DIR_NAMES or lower.endswith(".egg-info")


MAX_CHUNK_CHARS = 4000  # guvenlik siniri - asagidaki not'a bak


def chunk_code_file(text: str, rel_path: str, lines_per_chunk: int = LINES_PER_CHUNK,
                     overlap: int = OVERLAP_LINES) -> list:
    """Bir kod dosyasini satir penceresiyle boler, her parcaya dosya yolunu ekler."""
    lines = text.splitlines()
    if not lines:
        return []

    chunks = []
    step = max(lines_per_chunk - overlap, 1)
    for start in range(0, len(lines), step):
        window = lines[start:start + lines_per_chunk]
        if not window:
            continue
        body = "\n".join(window).strip()
        if not body:
            continue
        # Guvenlik siniri: bazi uretilmis dosyalarda (orn. SVD/register tanim
        # dosyalari) tek bir satir binlerce karakter olabiliyor - "80 satir"
        # kurali boyle durumda cok uzun bir chunk uretip embedding modelinde
        # GPU OOM'a yol acabiliyor (gozlemlendi). Karakter bazinda da kes.
        body = body[:MAX_CHUNK_CHARS]
        chunks.append(f"# Dosya: {rel_path}\n{body}")
        if start + lines_per_chunk >= len(lines):
            break
    return chunks


def chunk_doc_file(text: str, rel_path: str, max_chars: int = 1800) -> list:
    """
    Dokumantasyon/duz metin dosyalarini boler.

    .md dosyalari icin kitaplardaki ile AYNI baslik-farkinda chunking
    (chunk_by_headers) kullanilir - ornegin d2l-en reposu neredeyse tamamen
    basliklandirilmis markdown (kitap bolumleri gibi), bu yuzden kaba
    karakter-penceresi yerine baslik sinirlarina saygili bolme cok daha
    iyi retrieval kalitesi verir (bkz. Faz 1'deki kitap chunking notu).
    .rst/.txt gibi farkli baslik sozdizimine sahip formatlar icin basit
    karakter penceresi kullanmaya devam ediyoruz.
    """
    if rel_path.lower().endswith(".md"):
        raw_chunks = chunk_by_headers(text, max_chars=max_chars)
        return [f"# Dosya: {rel_path}\n{c}" for c in raw_chunks]

    chunks = []
    for i in range(0, len(text), max_chars):
        piece = text[i:i + max_chars].strip()
        if piece:
            chunks.append(f"# Dosya: {rel_path}\n{piece}")
    return chunks


def walk_repo(repo_path: str, repo_name: str, verbose: bool = True) -> list:
    """
    Bir repo klasorunu gezer, uygun dosyalari chunk'lar.

    Icerik-hash'ine gore tekrar eden dosyalari eler: ozellikle gomulu sistem
    ornek repolarinda (STM32 vb.) ayni HAL/CMSIS surucu dosyalari onlarca
    farkli ornek proje klasorunde birebir kopyalanmis olabiliyor. Ayni
    icerigi defalarca indekslemek hem index'i sisirir hem retrieval
    kalitesini dusurur (ayni chunk defalarca eslesir). Her benzersiz icerik
    sadece ILK gordugu yerde bir kez chunk'lanir.

    Donus: [{"text": ..., "source": repo_name, "file_path": rel_path}, ...]
    """
    results = []
    seen_hashes = set()
    skipped_duplicate = 0
    skipped_files = 0

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not should_skip_dir(d)]

        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, repo_path).replace("\\", "/")

            if ext not in CODE_EXTENSIONS and ext not in DOC_EXTENSIONS:
                continue

            try:
                with open(full_path, encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except OSError:
                skipped_files += 1
                continue

            if not text.strip():
                continue

            content_hash = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
            if content_hash in seen_hashes:
                skipped_duplicate += 1
                continue
            seen_hashes.add(content_hash)

            if ext in DOC_EXTENSIONS:
                chunks = chunk_doc_file(text, rel_path)
            else:
                chunks = chunk_code_file(text, rel_path)

            for c in chunks:
                results.append({"text": c, "source": repo_name, "file_path": rel_path})

    if verbose:
        print(f"  {repo_name}: {len(results)} chunk, "
              f"{skipped_duplicate} tekrar eden dosya elendi, "
              f"{skipped_files} dosya okunamadi")

    return results

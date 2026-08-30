"""
ADIM 1: PDF -> Markdown donusumu.

data/pdfs/ altindaki her PDF'i okunabilir markdown metnine cevirir ve
data/markdown/ altina yazar. pymupdf4llm kullaniliyor cunku duz metin
cikarmanin otesinde baslik hiyerarsisini (# ## ###) buyuk olcude koruyor -
bu da bir sonraki adimdaki (chunking.py) baslik-farkinda parcalamanin
calisabilmesi icin sart.

Zaten islenmis bir kitabi atlar (tekrar calistirmak guvenli).

Kullanim:
    python convert_pdfs.py
"""

import os
import glob
import pymupdf4llm

from config import PDF_SOURCE_DIR, MD_OUTPUT_DIR


def main():
    os.makedirs(MD_OUTPUT_DIR, exist_ok=True)
    pdf_files = glob.glob(os.path.join(PDF_SOURCE_DIR, "*.pdf"))

    if not pdf_files:
        print(f"UYARI: {PDF_SOURCE_DIR} icinde hic PDF bulunamadi.")
        return

    print(f"{len(pdf_files)} PDF bulundu:")
    for p in pdf_files:
        print(f"  - {os.path.basename(p)}")
    print()

    for pdf_path in pdf_files:
        name = os.path.splitext(os.path.basename(pdf_path))[0]
        out_path = os.path.join(MD_OUTPUT_DIR, f"{name}.md")

        if os.path.exists(out_path):
            print(f"[atlandi - zaten var] {name}")
            continue

        print(f"[isleniyor] {name} ...")
        try:
            md_text = pymupdf4llm.to_markdown(pdf_path)
        except Exception as e:
            print(f"  HATA: {e}")
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md_text)

        char_count = len(md_text)
        header_count = md_text.count("\n#")
        preview = md_text[:300].replace("\n", " ")

        print(f"  -> {char_count} karakter, ~{header_count} baslik bulundu")
        print(f"  -> onizleme: {preview}")

        # Taranmis (image-based) PDF'lerde metin katmani olmadigi icin
        # cikti neredeyse bos gelir - bu durumda OCR gerekir (ocrmypdf vb.).
        if char_count < 2000:
            print("  !! UYARI: cok kisa metin, PDF taranmis (image-based) olabilir, OCR gerekebilir.")
        print()


if __name__ == "__main__":
    main()

from pathlib import Path
import json

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_DIR = Path("data/pdfs")


def load_pdf(pdf_path: Path):
    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")

        if text.strip():
            pages.append({
                "text": text,
                "page": page_number
            })

    document.close()

    return pages



def create_chunks(pages, source_name):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []

    for page in pages:

        page_chunks = splitter.split_text(page["text"])

        for chunk_index, chunk in enumerate(page_chunks):

            chunks.append({
                "text": chunk,
                "source": source_name,
                "page": page["page"],
                "chunk_index": chunk_index
            })

    return chunks



if __name__ == "__main__":

    all_chunks = []

    pdf_files = list(PDF_DIR.glob("*.pdf"))

    print(f"PDFs found: {len(pdf_files)}")

    for pdf_path in pdf_files:

        print(f"\nProcessing {pdf_path.name}")

        pages = load_pdf(pdf_path)

        chunks = create_chunks(pages, pdf_path.name)

        print(f"Pages: {len(pages)}")
        print(f"Chunks: {len(chunks)}")

        all_chunks.extend(chunks)

    output_path = Path("data/processed/chunks.json")

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(all_chunks, file, ensure_ascii=False, indent=2)

    print("\nFinished!")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Saved to: {output_path}")
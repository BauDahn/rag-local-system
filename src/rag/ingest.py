# src/rag/ingest.py

import hashlib
import os
from typing import List
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_documents(
    raw_data_path: str, chunk_size: int, chunk_overlap: int, min_letters: int = 120
) -> List[Document]:
    """
    Reads every PDF in a directory, extracts text page by page and splits it into
    chunks. Each chunk keeps its source file and page number as metadata so answers
    can cite where they came from.
    """
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"The specified path does not exist: {raw_data_path}")

    pdf_files = sorted(f for f in os.listdir(raw_data_path) if f.lower().endswith(".pdf"))
    if not pdf_files:
        print(f"[Warning] No PDF files found in {raw_data_path}. Please add documents to parse.")
        return []

    pages: List[Document] = []
    for pdf_file in pdf_files:
        print(f"[Ingest] Extracting text from: {pdf_file}")
        try:
            reader = PdfReader(os.path.join(raw_data_path, pdf_file))
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    pages.append(Document(page_content=text, metadata={"source": pdf_file, "page": page_num}))
        except Exception as e:
            print(f"[Error] Failed to read {pdf_file}: {e}")

    if not pages:
        print("[Warning] Total extracted text is empty.")
        return []

    # Splitting per page keeps chunks from crossing document boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    # Drop near-empty chunks (title pages, page numbers, "!" separators): they carry
    # no information but can still win a similarity search.
    before = len(chunks)
    chunks = [c for c in chunks if sum(ch.isalpha() for ch in c.page_content) >= min_letters]
    print(f"[Ingest] Dropped {before - len(chunks)} low-content chunks.")
    print(f"[Ingest] Generated {len(chunks)} chunks from {len(pages)} pages in {len(pdf_files)} file(s).")
    return chunks


def hash_pdfs(raw_data_path: str) -> str:
    """Fingerprint of the PDFs in a directory (names + contents), stored in the index manifest."""
    digest = hashlib.sha256()
    for pdf_file in sorted(f for f in os.listdir(raw_data_path) if f.lower().endswith(".pdf")):
        digest.update(pdf_file.encode())
        with open(os.path.join(raw_data_path, pdf_file), "rb") as f:
            digest.update(f.read())
    return digest.hexdigest()

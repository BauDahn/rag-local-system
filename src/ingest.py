import os
from typing import List
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_documents(raw_data_path: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Scrapes a local directory for PDF files, extracts their raw text, 
    and splits them into structured chunks based on character constraints.
    """
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"The specified path does not exist: {raw_data_path}")
        
    pdf_files = [f for f in os.listdir(raw_data_path) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"[Warning] No PDF files found in {raw_data_path}. Please add documents to parse.")
        return []

    combined_text = ""
    
    # 1. Scrape and extract text from all available PDFs
    for pdf_file in pdf_files:
        file_path = os.path.join(raw_data_path, pdf_file)
        print(f"[Ingest] Extracting text from: {pdf_file}")
        
        try:
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    combined_text += page_text + "\n"
        except Exception as e:
            print(f"[Error] Failed to read {pdf_file}: {str(e)}")

    if not combined_text.strip():
        print("[Warning] Total extracted text is empty.")
        return []

    # 2. Initialize the scientific character splitter
    # It intelligently splits by paragraphs, sentences, and words to avoid cutting information in half
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    # 3. Execute chunking
    chunks = text_splitter.split_text(combined_text)
    print(f"[Ingest] Successfully generated {len(chunks)} text chunks.")
    
    return chunks

if __name__ == "__main__":
    # Quick isolated module test using default fallbacks
    print("Testing Ingest Module locally...")
    import yaml
    
    # Load configuration parameters
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    try:
        sample_chunks = load_and_split_documents(
            raw_data_path=config["paths"]["raw_data"],
            chunk_size=config["parameters"]["chunk_size"],
            chunk_overlap=config["parameters"]["chunk_overlap"]
        )
        if sample_chunks:
            print("\n--- Sample Chunk Preview ---")
            print(sample_chunks[0])
            print("----------------------------")
    except Exception as e:
        print(f"Test failed: {e}")
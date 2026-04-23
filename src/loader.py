from pypdf import PdfReader
import os

def load_pdf(file_path):
    """
    Reads a PDF and returns extracted text.
    """
    reader = PdfReader(file_path)
    
    full_text = ""
    total_pages = len(reader.pages)
    file_name = os.path.basename(file_path)
    print(f"Processing {file_name} with {total_pages} pages...")
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += f"\n--- Page {page_num + 1} ---\n"
            full_text += text
    
    return full_text
from pypdf import PdfReader
import re

def load_pdf(path):
    reader = PdfReader(path)
    chunks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            # simple split into smaller chunks
            paragraphs = re.split(r"\n{2,}", text)
            for para in paragraphs:
                cleaned = para.strip().replace("\n", " ")
                if len(cleaned) > 30:  # skip very short lines
                    chunks.append({
                        "text": cleaned,
                        "page": i + 1
                    })
    return chunks

def load_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    paragraphs = re.split(r"\n{2,}", text)
    chunks = []
    for i, para in enumerate(paragraphs):
        cleaned = para.strip().replace("\n", " ")
        if len(cleaned) > 30:
            chunks.append({
                "text": cleaned,
                "page": 1  # no real pages in TXT
            })
    return chunks


if __name__ == "__main__":
    chunks = load_pdf("data/www-kaggle-com-doc....pdf")
    for c in chunks:
        print(c)
        print("===================")

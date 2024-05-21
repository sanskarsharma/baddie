import random
import string
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader


def generate_random_string(*, length=10):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        chunk_size=512,
        chunk_overlap=103,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

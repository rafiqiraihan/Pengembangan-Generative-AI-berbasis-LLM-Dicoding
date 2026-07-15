import os
import torch
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Perangkat yang digunakan: {device}")

# Load Document
file_path = r"./buku_panduan_gen_ai.pdf"
loader = PyMuPDFLoader(file_path)
documents = loader.load()

print(documents[0:7])

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", "."]
)

chunks = text_splitter.split_documents(documents)
print(f"Dokumen dipecah menjadi {len(chunks)} chunks")
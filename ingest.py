from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import os

print("=" * 60)
print("🚀 AI RETRIEVAL ASSISTANT")
print("=" * 60)

documents = []

folder = "documents"

print("\n📂 Scanning documents folder...\n")

for file in os.listdir(folder):

    path = os.path.join(folder, file)

    try:

        if file.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            docs = loader.load()

        elif file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs = loader.load()

        elif file.endswith(".docx"):
            loader = Docx2txtLoader(path)
            docs = loader.load()

        else:
            continue

        documents.extend(docs)

        print(f"✅ Loaded: {file}")

    except Exception as e:

        print(f"❌ Error loading {file}")
        print(e)

print("\n-----------------------------------")
print(f"Total Documents : {len(documents)}")
print("-----------------------------------")

if len(documents) == 0:

    print("\nNo documents found!")

    exit()
    print("\n✂ Splitting documents into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

docs = splitter.split_documents(documents)

print(f"✅ Created {len(docs)} chunks")

print("\n🧠 Loading HuggingFace Embedding Model...")

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("✅ Embedding model loaded")

print("\n📦 Creating FAISS Vector Database...")

db = FAISS.from_documents(
    docs,
    embedding
)

db.save_local("vectorstore")

print("\n" + "=" * 60)
print("🎉 Vector Database Created Successfully!")
print("=" * 60)

print("\nSaved to: vectorstore/")
print(f"Documents : {len(documents)}")
print(f"Chunks    : {len(docs)}")
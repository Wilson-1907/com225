import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ==============================
# PDF FOLDER
# ==============================

PDFS_FOLDER = "pdfs"

# Create folder if it doesn't exist
os.makedirs(PDFS_FOLDER, exist_ok=True)

print("\n========== COM225 PDF INGESTION ==========\n")

# Find all PDFs
pdf_files = [f for f in os.listdir(PDFS_FOLDER) if f.endswith(".pdf")]

if not pdf_files:
    print("⚠️ No PDF files found!")
    print(f"Please add PDF files inside the '{PDFS_FOLDER}' folder.")
    exit(1)

print(f"Found {len(pdf_files)} PDF files.\n")

# ==============================
# LOAD PDFs
# ==============================

all_docs = []

for file in pdf_files:

    try:
        file_path = os.path.join(PDFS_FOLDER, file)

        print(f"Loading: {file}")

        loader = PyPDFLoader(file_path)

        docs = loader.load()

        all_docs.extend(docs)

        print(f"✓ SUCCESS: {file}")

    except Exception as e:

        print(f"✗ FAILED: {file}")
        print(f"ERROR: {e}\n")

# ==============================
# SPLIT DOCUMENTS
# ==============================

print("\n========== SPLITTING DOCUMENTS ==========\n")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

documents = splitter.split_documents(all_docs)

print(f"Total chunks created: {len(documents)}")

# ==============================
# EMBEDDINGS
# ==============================

print("\n========== CREATING EMBEDDINGS ==========\n")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# ==============================
# CREATE VECTOR DATABASE
# ==============================

vectorstore = FAISS.from_documents(documents, embeddings)

vectorstore.save_local("vectorstore")

print("\n✅ SUCCESS!")
print("Vector database created successfully.")
print("Saved in: vectorstore/")


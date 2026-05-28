
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

PDF_FOLDER = "pdfs"

all_documents = []

print("\n========== COM225 INGESTION ==========\n")

pdf_files = [
    file for file in os.listdir(PDF_FOLDER)
    if file.endswith(".pdf")
]

print(f"Found {len(pdf_files)} PDF files.\n")

for file in pdf_files:

    try:

        pdf_path = os.path.join(PDF_FOLDER, file)

        print(f"Loading: {file}")

        loader = PyPDFLoader(pdf_path)

        documents = loader.load()

        for doc in documents:
            doc.metadata["source_file"] = file

        all_documents.extend(documents)

        print(f"SUCCESS: {file}")

    except Exception as e:

        print(f"FAILED: {file}")
        print(f"ERROR: {e}\n")

print("\n========== SPLITTING DOCUMENTS ==========\n")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(all_documents)

print(f"Total chunks created: {len(chunks)}")

print("\n========== CREATING EMBEDDINGS ==========\n")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

vectorstore.save_local("vectorstore")

print("\n========== SUCCESS ==========")
print("Vector database created successfully!")
print("Saved in: vectorstore/")


import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Create lectures folder if it doesn't exist
LECTURES_FOLDER = "lectures"
os.makedirs(LECTURES_FOLDER, exist_ok=True)

print("\n========== COM225 LECTURE INGESTION ==========\n")

# Read all text files from lectures folder
all_texts = []
text_files = [f for f in os.listdir(LECTURES_FOLDER) if f.endswith(('.txt', '.md'))]

if not text_files:
    print("⚠️  No lecture files found!")
    print(f"Please add .txt or .md files to the '{LECTURES_FOLDER}' folder")
    print("\nExample file format:")
    print("-" * 50)
    print("File: linux_commands.txt")
    print("Content:")
    print("Linux Commands:")
    print("ls - list directory contents")
    print("cd - change directory")
    print("pwd - print working directory")
    print("-" * 50)
    exit(1)

print(f"Found {len(text_files)} lecture files.\n")

for file in text_files:
    try:
        file_path = os.path.join(LECTURES_FOLDER, file)
        print(f"Loading: {file}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        all_texts.append({
            "content": content,
            "source": file
        })
        print(f"✓ Loaded {file}")
        
    except Exception as e:
        print(f"✗ Failed: {file}")
        print(f"  Error: {e}\n")

print("\n========== SPLITTING INTO CHUNKS ==========\n")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Smaller chunks = less memory
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

chunks = []
for text_data in all_texts:
    text_chunks = splitter.split_text(text_data["content"])
    for chunk in text_chunks:
        chunks.append({
            "page_content": chunk,
            "metadata": {"source_file": text_data["source"]}
        })

print(f"Total chunks created: {len(chunks)}")

print("\n========== CREATING EMBEDDINGS ==========\n")
print("Loading lightweight embedding model...")

# Use the smallest possible embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Convert to FAISS format
from langchain.schema import Document
documents = [
    Document(page_content=chunk["page_content"], metadata=chunk["metadata"])
    for chunk in chunks
]

vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("vectorstore")

print("\n✅ SUCCESS!")
print(f"Vector database created from {len(text_files)} lecture files")
print("Saved in: vectorstore/")
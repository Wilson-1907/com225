import os
from dotenv import load_dotenv
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import gc

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Memory optimization
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

print("Loading embeddings model (lightweight)...")

# Load embeddings with minimal memory
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

print("Loading vector database...")

# Load vectorstore
vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

# Free up some memory
gc.collect()

print("✅ Chatbot ready!\n")

system_prompt = """
You are COM225 AI Assistant. You help university students revise COM225.

Your specialization includes:
- Linux Operating Systems
- Open Source Systems
- Linux Commands
- Full Stack Development
- OSS Concepts

Rules:
1. Use ONLY the provided lecture materials
2. Give clear, concise explanations
3. Use examples where possible
4. Format Linux commands properly using code blocks
5. If answer is missing, say: "This information is not in the current lecture materials"
6. Always teach like a friendly tutor
7. Keep responses educational but not overly long
"""

def ask_question(question):
    # Search for relevant content
    docs = vectorstore.similarity_search(question, k=3)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    sources = list(set([doc.metadata.get("source_file", "Unknown") for doc in docs]))
    
    final_prompt = f"""
Use the context below to answer the question.

CONTEXT:
{context}

QUESTION:
{question}

Instructions:
- If the context doesn't contain the answer, say so clearly
- Use examples from the context when possible
- For Linux commands, show the command and explain what it does
"""

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt}
        ],
        temperature=0.3,
        max_tokens=800  # Reduced for faster response
    )
    
    answer = response.choices[0].message.content
    
    formatted_sources = "\n".join([f"📚 {source}" for source in sources])
    
    final_answer = f"""
{answer}

---
**Sources:**  
{formatted_sources}
"""
    
    return final_answer
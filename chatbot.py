import os

from dotenv import load_dotenv
from groq import Groq

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

print("Loading embeddings model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading vector database...")

vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

print("Chatbot ready!\n")

system_prompt = """
You are COM225 AI Assistant.

You help university students revise COM225.

Your specialization includes:
- Linux Operating Systems
- Open Source Systems
- Linux Commands
- Full Stack Development
- OSS Concepts

Rules:
1. Use only the uploaded study materials.
2. Give clear explanations.
3. Use examples where possible.
4. Be concise but educational.
5. If answer is missing from notes, say:
   'I could not find this in the uploaded materials.'
6. If asked Linux commands, format them properly.
7. Always teach like a tutor.
"""

def ask_question(question):

    docs = vectorstore.similarity_search(
        question,
        k=4
    )

    context = "\n\n".join([
        doc.page_content for doc in docs
    ])

    sources = list(set([
        doc.metadata.get("source_file", "Unknown Source")
        for doc in docs
    ]))

    final_prompt = f"""
Use the context below to answer the question.

CONTEXT:
{context}

QUESTION:
{question}
"""

    response = client.chat.completions.create(

        model="llama3-70b-8192",

        messages=[

            {
                "role": "system",
                "content": system_prompt
            },

            {
                "role": "user",
                "content": final_prompt
            }

        ],

        temperature=0.3,

        max_tokens=1024
    )

    answer = response.choices[0].message.content

    formatted_sources = "\n".join([
        f"- {source}" for source in sources
    ])

    final_answer = f"""
{answer}

----------------------------

Sources:
{formatted_sources}
"""

    return final_answer

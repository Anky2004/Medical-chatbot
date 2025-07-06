# app.py
# app.py
from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_together import ChatTogether
from langchain_core.prompts import ChatPromptTemplate



# ---------- ENV / SETTINGS ----------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or "your_pinecone_key"
INDEX_NAME = "medical-chatbot"
PINECONE_REGION = "us-east-1"
DATA_DIR = "data/"
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

app = Flask(__name__, template_folder="templates", static_folder="static")
# CORS(app)  # Uncomment if needed

# ---------- Embeddings ----------
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------- Pinecone Setup ----------
pc = Pinecone(api_key=PINECONE_API_KEY)
index_names = [i.name for i in pc.list_indexes()]

if INDEX_NAME not in index_names:
    loader = DirectoryLoader(DATA_DIR, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    texts = splitter.split_documents(docs)

    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
    )

    PineconeVectorStore.from_documents(texts, embedding=embedding, index_name=INDEX_NAME)

vectorstore = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embedding)
retriever = vectorstore.as_retriever()

# ---------- LLM (Together AI) ----------
llm = ChatTogether(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    api_key=TOGETHER_API_KEY,
    temperature=0.4,
    max_tokens=512
)

# ---------- Prompt Template ----------
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "address it like a chat-bot and always greet nicely"
    "If you don't know the answer, say you don't know. "
    "Use three sentences maximum and keep the answer concise.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ---------- Routes ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("message")

    if not query:
        return jsonify({"error": "No input provided"}), 400

    try:
        result = rag_chain.invoke({"input": query})
        reply = result.get("answer", "I'm not sure how to respond to that.") if isinstance(result, dict) else str(result)

        print("User Query:", query)
        print("Bot Reply:", reply)

        return jsonify({"reply": reply})
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "Something went wrong processing your request."}), 500


if __name__ == "__main__":
    app.run(debug=True)
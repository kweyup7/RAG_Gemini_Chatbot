import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, session
import google.generativeai as genai
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load documents
loader = DirectoryLoader("docs/", glob="*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(chunks, embeddings)

# LLM + Conversational chain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever
)

# Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

@app.route("/", methods=["GET", "POST"])
def index():
    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        user_query = request.form["query"]

        # Run through LangChain Conversational Retrieval
        result = qa_chain(
            {"question": user_query, "chat_history": [(m["role"], m["text"]) for m in session["chat_history"]]}
        )
        answer = result["answer"]

        # Save conversation
        session["chat_history"].append({"role": "user", "text": user_query})
        session["chat_history"].append({"role": "bot", "text": answer})
        session.modified = True

    return render_template("index.html", chat_history=session["chat_history"])

if __name__ == "__main__":
    app.run(debug=True)

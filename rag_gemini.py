import os
from dotenv import load_dotenv
from flask import Flask, render_template, request
import google.generativeai as genai
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


# 1. Load API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# 2. Load documents (PDFs from docs/ folder)
loader = DirectoryLoader("docs/", glob="*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

# 3. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# 4. Create embeddings using Gemini
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5. Use Gemini for the chat model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")  # or "gemini-pro"
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    answer = None
    if request.method == "POST":
        query = request.form["query"]
        answer = qa.run(query)
    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)

# RAG Chatbot with Gemini + Flask

This project is a simple **Retrieval-Augmented Generation (RAG) chatbot** built with:
- [Google Gemini](https://ai.google.dev/) for LLM responses  
- [LangChain](https://www.langchain.com/) for retrieval & orchestration  
- [FAISS](https://github.com/facebookresearch/faiss) for vector search  
- [Flask](https://flask.palletsprojects.com/) for the web app  

The bot can **answer questions about your local documents** (PDF, TXT, etc.) by combining document search with LLM reasoning.

---

##  Features
- Ask questions in natural language
- RAG-powered responses (retrieves context + generates answers)
- Simple web interface using Flask
  
---

##  Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/rag-chatbot-gemini.git
   cd rag_bot_gemini
   
2. **Create a Virtual environment**
   ```bash
   python -m venv venv
    source venv/bin/activate   # Mac/Linux
    venv\Scripts\activate      # Windows

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

4. **Set up environment variables in the .env file**
   ```env
   GOOGLE_API_KEY=your_google_api_key_here

## Running the APP

5. **Start the Flask server**
   ```bash
   python rag_gemini.py
   
6. **Open your browser and visit:**
    ```cpp
    http://127.0.0.1:5000

# LangChain RAG — PDF Document Q&A

A simple Retrieval-Augmented Generation (RAG) pipeline that lets you ask natural-language questions over your own PDF documents. Built with **LangChain**, **ChromaDB**, **HuggingFace embeddings**, and **Groq** for fast LLM inference.

## Overview

This project is split into two notebooks that together form the RAG pipeline:

1. **`langchain_rag_doc_ingestion.ipynb`** — Loads PDF documents, splits them into chunks, generates embeddings, and stores them in a persistent Chroma vector database.
2. **`langchain_rag_retrieval.ipynb`** — Loads the existing vector database, retrieves the most relevant chunks for a user query, and generates an answer using a Groq-hosted LLM.

## How It Works

**Ingestion pipeline:**
```
PDF documents → DirectoryLoader/UnstructuredFileLoader → CharacterTextSplitter
             → HuggingFace Embeddings → Chroma vector store (persisted to disk)
```

**Retrieval pipeline:**
```
User query → Chroma retriever → relevant chunks → RetrievalQA chain (Groq LLM)
          → generated answer + source documents
```

## Tech Stack

- **LangChain** (`langchain-community`, `langchain-classic`, `langchain-text-splitters`) — orchestration
- **ChromaDB** / `langchain-chroma` — vector store
- **HuggingFace** (`langchain-huggingface`, `sentence-transformers`) — text embeddings
- **Groq** (`langchain-groq`) — fast LLM inference for answer generation
- **Unstructured** (`unstructured`, `unstructured[pdf]`) — PDF parsing
- **NLTK** — tokenization support for document processing

## Project Structure

```
.
├── langchain_rag_doc_ingestion.ipynb   # Step 1: build the vector database from PDFs
├── langchain_rag_retrieval.ipynb       # Step 2: query the vector database
├── requirements.txt
├── docs/                               # Place your source PDF files here
└── vector_db/                          # Generated Chroma vector store (created on first run)
```

## Setup

### Prerequisites

- Python 3.11+ (see note below on Python 3.14 compatibility)
- A [Groq API key](https://console.groq.com/keys)

### Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd <your-repo-name>

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root (this file is git-ignored and should never be committed):

```
GROQ_API_KEY=your_groq_api_key_here
```

Load it in your notebook with:

```python
from dotenv import load_dotenv
load_dotenv()
```

> **Never hardcode API keys directly in notebook cells.** Once committed to Git history, a key remains exposed even if you delete it in a later commit — you must revoke and regenerate it.

## Usage

1. Place your PDF files in a `docs/` folder in the project root.
2. Open and run **`langchain_rag_doc_ingestion.ipynb`** top to bottom. This will:
   - Load and chunk your PDFs
   - Generate embeddings
   - Persist a Chroma vector store to `vector_db/`
3. Open and run **`langchain_rag_retrieval.ipynb`** top to bottom. This will:
   - Load the persisted vector store
   - Initialize the Groq LLM
   - Let you query your documents, e.g.:
     ```python
     query = "What does the document say about Evolution and Ecosystem?"
     response = qa_chain.invoke({"query": query})
     print(response["result"])
     ```
   - Return both a generated answer and the source document chunks used

## Configuration

Key parameters you can tune in the notebooks:

| Parameter | Location | Description |
|---|---|---|
| `chunk_size` | ingestion notebook | Size of each text chunk (default: 2000) |
| `chunk_overlap` | ingestion notebook | Overlap between chunks (default: 500) |
| `model` | retrieval notebook | Groq model used for generation (e.g. `openai/gpt-oss-20b`) |
| `temperature` | retrieval notebook | LLM sampling temperature (default: 0.0 for deterministic answers) |

## Notes

- Groq's model lineup changes over time — if you hit a `model_decommissioned` error, check the [Groq models page](https://console.groq.com/docs/models) for the current recommended replacement.
- This project uses `langchain-classic` for `RetrievalQA`, since that chain was moved out of core `langchain` in recent versions (1.x+).
- If running on Python 3.14, some dependencies (e.g. `tiktoken`) may not have prebuilt wheels yet and can fail to install. If you hit build errors, use Python 3.11 or 3.12 instead.

## License

Add your license of choice here (e.g. MIT).

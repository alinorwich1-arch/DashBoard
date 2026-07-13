# YCE AI Dashboard — RAG-Powered Document Intelligence

A Streamlit application that combines:
- **📄 RAG Document Search** — Upload PDFs to Pinecone and ask questions using intent-aware retrieval
- **💬 Gemini Chatbot** — Conversational AI with optional document grounding

## Features
- Intent-aware retrieval (`summarize`, `table of contents`, `section lookup`, `general Q&A`)
- Fast single-call document summarization via broad keyword sweep
- HyDE (Hypothetical Document Embeddings) with result caching
- Parallel Pinecone queries via ThreadPoolExecutor
- Aurora Midnight / Synthwave UI theme
- Supports any text-based PDF: financial, legal, HR, technical, medical, academic

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure secrets
Create `.streamlit/secrets.toml`:
```toml
PINECONE_API_KEY = "your_pinecone_key"
GEMINI_API_KEY   = "your_gemini_key"
PINECONE_INDEX   = "yce"
```

### 4. Run locally
```bash
streamlit run app.py
```

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (secrets.toml is gitignored — never committed)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set `app.py` as the main file
4. Click **Advanced settings → Secrets** and paste:
```
PINECONE_API_KEY = "your_pinecone_key"
GEMINI_API_KEY   = "your_gemini_key"
PINECONE_INDEX   = "yce"
```
5. Click **Deploy** — done!

## Tech Stack
- [Streamlit](https://streamlit.io)
- [LangChain](https://langchain.com)
- [Pinecone](https://pinecone.io)
- [Google Gemini](https://ai.google.dev)
- [HuggingFace Sentence Transformers](https://huggingface.co) (`all-MiniLM-L6-v2`)

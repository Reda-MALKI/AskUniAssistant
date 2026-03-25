# AskUniAssistant
AskUniAssistant

# RAG ON DOCUMENTS – AI Chatbot for University Regulations (University of Louis Pasteur)

## Overview

**RAG On Documents** is an intelligent AI-powered chatbot that allows users to query
university regulation documents using natural language.
Instead of manually reading through long PDF files, users can simply ask questions and
get accurate, context-aware answers instantly.

Built using **Retrieval-Augmented Generation (RAG)**, the system retrieves the most
relevant sections from the document before generating a precise response —
reducing hallucinations and improving reliability.

---

## What It Does

- Loads and parses a university regulation PDF
- Cleans and chunks the document by chapters and sections
- Embeds the chunks using OpenAI Embeddings and stores them in a FAISS vector database
- On each question, retrieves the top 4 most relevant chunks
- Sends them as context to GPT-4o-mini to generate a grounded answer
- Exposes everything through a clean Flask web interface

---

## Project Structure
```
RAG_ON_DOCUMENTS/
│
├── data/
│   └── University Luis and Pasteur Reglementations.pdf
├── static/
├── templates/
│   └── index.html
├── app.py
└── requirements.txt
```

---

## How It Works (RAG Pipeline)
```
User Question
      ↓
Query sent to FAISS Vector Store
      ↓
Top 4 relevant chunks retrieved
      ↓
Chunks injected as context into the prompt
      ↓
GPT-4o-mini generates a grounded answer
      ↓
Answer returned to the user via Flask API
```

---

## Tech Stack

### Backend
- Python
- Flask (REST API)

### AI / LLM
- OpenRouter API
- GPT-4o-mini (generation)
- text-embedding-3-small (embeddings)

### RAG & Vector Search
- LangChain
- FAISS (Facebook AI Similarity Search)

### Document Processing
- PyPDF2 (PDF extraction)
- Custom chunking by chapter & section (Regex-based)

---

## Key Features

- **Smart PDF Chunking** — splits the document by `CHAPITRE` and numbered sections for precise retrieval
- **Context-Grounded Answers** — the LLM only answers using retrieved content, never guesses
- **Hallucination Control** — if the answer is not in the document, the model says *"I don't know"*
- **REST API** — clean `/ask` endpoint accepting JSON input
- **Web Interface** — simple frontend via `index.html`

---

## Environment Setup

Set your API key:
```bash
setx OPENROUTER_API_KEY "your_api_key"
```

---

## Run the Project
```bash
pip install -r requirements.txt
python app.py
```

Open: http://127.0.0.1:5000

---

## Example Usage

**Question:**
> "What are the conditions to validate a semester?"

**Answer:**
> "According to Chapter 3, Section 3.2, a semester is validated when the student
obtains an average of at least 10/20 across all course units..."

---

## Important – Why RAG?

Standard LLMs have no knowledge of your specific documents.
RAG solves this by:

- Embedding your document into a searchable vector space
- Retrieving only the relevant sections at query time
- Feeding them as context to the LLM for accurate, document-based answers

This makes the system **reliable, explainable, and document-faithful**.

# 📚 Smart Multi-PDF AI Chatbot (Offline)

An offline AI assistant that reads multiple PDF files, understands their content, and answers questions or generates summaries using NLP.

---

## 🚀 Features

- Upload and read multiple PDFs
- Ask questions based on document content
- Automatic document summarization
- Semantic search using embeddings
- Works 100% offline (no API needed)
- Fast and lightweight Streamlit interface

---

## 🧠 How it works

1. Extracts text from PDFs (PyPDF)
2. Splits text into overlapping chunks
3. Converts chunks into embeddings using Sentence Transformers
4. Finds the most relevant chunks using cosine similarity
5. Generates answers & summaries using a HuggingFace model
6. Displays results with Streamlit

---

## 🛠️ Tech Stack

- Python
- Streamlit
- PyPDF
- Sentence Transformers (MiniLM)
- HuggingFace Transformers
- NumPy
- Torch

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py

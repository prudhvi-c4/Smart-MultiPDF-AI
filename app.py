import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
from numpy.linalg import norm
import hashlib
import re

st.set_page_config(page_title="Smart Multi-PDF AI Chatbot")
st.title("📚 Smart Multi-PDF AI Chatbot (Offline)")

# ------------------------
# Model loading (cached)
# ------------------------


@st.cache_resource
def load_models():
    # Embedding model (small & fast)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # ONE small generative model (used for both Q&A + summary)
    gen_model = pipeline(
        "text2text-generation",
        model="google/flan-t5-small"
    )

    return embedder, gen_model


embedder, gen_model = load_models()


# ------------------------
# Helpers
# ------------------------
def get_file_hash(uploaded_files):
    """Unique key for a set of uploaded files (for caching)."""
    h = hashlib.md5()
    for f in uploaded_files:
        h.update(f.name.encode())
    return h.hexdigest()


def split_text(text, chunk_size=800, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


@st.cache_data(show_spinner=True)
def process_pdfs(file_key, uploaded_files, max_pages=5):
    """Read PDFs, limit pages for speed, create chunks + embeddings."""
    all_chunks = []
    total_pages = 0

    for pdf in uploaded_files:
        reader = PdfReader(pdf)
        num_pages = len(reader.pages)
        total_pages += num_pages

        text = ""
        for i, page in enumerate(reader.pages):
            if i >= max_pages:
                break
            page_text = page.extract_text()
            if page_text:
                text += " " + page_text

        if text.strip():
            chunks = split_text(text)
            all_chunks.extend(chunks)

    if not all_chunks:
        return [], None, total_pages

    embeddings = embedder.encode(
        all_chunks, convert_to_numpy=True, show_progress_bar=True)
    norms = norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    return all_chunks, embeddings, total_pages


# ------------------------
# Chat memory
# ------------------------
if "history" not in st.session_state:
    st.session_state.history = []


st.write("Upload one or more PDFs and ask questions (100% offline, longer answers).")

uploaded_pdfs = st.file_uploader(
    "Upload your PDFs", type="pdf", accept_multiple_files=True)

if uploaded_pdfs:

    file_key = get_file_hash(uploaded_pdfs)

    with st.spinner("📖 Reading and indexing PDFs (first time only)..."):
        chunks, chunk_embeddings, total_pages = process_pdfs(
            file_key, uploaded_pdfs)

    st.sidebar.write(f"📄 Total pages in PDFs: {total_pages}")
    st.sidebar.write(f"🧩 Chunks created: {len(chunks)}")

    if not chunks:
        st.error(
            "No text could be read from the PDFs. Try a text-based PDF (not scanned).")
    else:
        with st.expander("🔍 Preview of extracted content"):
            st.write(chunks[0][:1000] + "...")

        question = st.text_input("Ask a question about your PDFs")

        col1, col2 = st.columns(2)

        # ------------------
        # DETAILED ANSWER
        # ------------------
        if col1.button("💬 Get Detailed Answer") and question:

            with st.spinner("🤖 Thinking..."):
                # 1) retrieve relevant chunks
                q_emb = embedder.encode([question], convert_to_numpy=True)[0]
                q_emb = q_emb / norm(q_emb)

                scores = np.dot(chunk_embeddings, q_emb)
                top_k = 3
                top_idx = np.argsort(scores)[-top_k:][::-1]
                context = " ".join(chunks[i] for i in top_idx)

                # 2) build prompt for generative model
                prompt = (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer in 4–6 sentences with a clear explanation, "
                    "using only the information in the context."
                )

                result = gen_model(
                    prompt,
                    max_new_tokens=256,
                    do_sample=False
                )[0]["generated_text"]

                answer = result.strip()

                # Save to memory
                st.session_state.history.append((question, answer))

                st.subheader("✅ Answer")
                st.write(answer)

        # ------------------
        # SUMMARY
        # ------------------
                # ------------------
        # SUMMARY (with de-dup)
        # ------------------
                # ------------------
        # SUMMARY (bullet style, de-duplicated)
        # ------------------
        if col2.button("📝 Summarize Document") and chunks:

            with st.spinner("📘 Creating summary..."):
                # use first few chunks as main context
                context = " ".join(chunks[:5])

                prompt = (
                    "Read the following text and write a clear summary as 3 to 5 bullet points. "
                    "Each bullet must describe a different idea. "
                    "Do not repeat the same sentence or phrase.\n\n"
                    f"TEXT:\n{context}"
                )

                raw = gen_model(
                    prompt,
                    max_new_tokens=256,
                    do_sample=False
                )[0]["generated_text"].strip()

                # split into lines, remove duplicates/empty
                lines = [l.strip("-• \n") for l in raw.split("\n")]
                bullets = []
                seen = set()
                for l in lines:
                    if not l:
                        continue
                    key = l.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    bullets.append(l)

                # keep max 5 bullets
                bullets = bullets[:5]

                st.subheader("📌 Summary")
                if not bullets:
                    st.write(
                        "Could not generate a clear summary for this document.")
                else:
                    for b in bullets:
                        st.markdown(f"- {b}")

        # ------------------
        # MEMORY DISPLAY
        # ------------------
        if st.session_state.history:
            st.subheader("🧠 Chat History (this session)")
            for i, (q, a) in enumerate(reversed(st.session_state.history)):
                st.markdown(f"**Q{i+1}: {q}**")
                st.markdown(f"A: {a}")
                st.markdown("---")

else:
    st.info("👆 Upload PDFs to begin.")

import streamlit as st 
import fitz
import sqlite3
import faiss
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

st.set_page_config(page_title="PDF Research Assistant", layout="wide")


@st.cache_resource(show_spinner=False)
def load_models():
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
        return summarizer, semantic_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

summarizer, semantic_model = load_models()
if not summarizer or not semantic_model:
    st.error("Failed to load models. Please check dependencies or restart the application.")

DB_PATH = "document_data.db"

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join([page.get_text("text") for page in doc])

def clean_and_segment_text(text):
    sections = {}
    current_section = "Introduction"
    for line in text.split("\n"):
        line = line.strip()
        if re.match(r'^[A-Z][a-z]+(\s[A-Z][a-z]+)*$', line) and len(line) < 50:
            current_section = line
            sections[current_section] = []
        else:
            sections.setdefault(current_section, []).append(line)
    return {k: "\n".join(v) for k, v in sections.items()}

def save_to_database(data):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS document_sections")
    cursor.execute("CREATE TABLE document_sections (section TEXT, content TEXT)")
    for section, content in data.items():
        cursor.execute("INSERT INTO document_sections (section, content) VALUES (?, ?)", (section, content))
    conn.commit()
    conn.close()

def summarize_text(text, length="short"):
    if not summarizer:
        return "Summarization model not available. Please check the setup."
    
    chunks = [text[i:i+1024] for i in range(0, min(len(text), 5000), 1024)] 
    summaries = []
    
    for chunk in chunks:
        try:
            summary = summarizer(chunk, max_length=150 if length == "detailed" else 60, 
                                 min_length=50 if length == "detailed" else 30, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            st.error(f"Error during summarization: {e}")
            return "Error generating summary."
        
        if length == "short" and len(summaries) >= 2:  
            break

    return "\n\n".join(summaries)

st.title("📄 PDF Research Assistant")

uploaded_file = st.file_uploader("📂 Upload a PDF", type="pdf")
if uploaded_file:
    with st.spinner("🔍 Processing PDF... Please wait..."):
        try:
            pdf_text = extract_text_from_pdf(uploaded_file)
            structured_data = clean_and_segment_text(pdf_text)
            save_to_database(structured_data)
            st.success("✅ PDF processed successfully!")
        except Exception as e:
            st.error(f"Error processing PDF: {e}")

if uploaded_file:
    summary_type = st.radio("📌 Summary Type", ["Short", "Detailed"])
    
    if st.button("📜 Generate Summary"):
        with st.spinner("📝 Summarizing... Please wait..."):
            summary = summarize_text(pdf_text, length="detailed" if summary_type == "Detailed" else "short")
        st.write("### 📝 Summary")
        st.write(summary)

st.markdown("## 📩 Contact Me")
st.markdown('[✉️ Send Email](mailto:dasdevanshi7@gmail.com)', unsafe_allow_html=True)
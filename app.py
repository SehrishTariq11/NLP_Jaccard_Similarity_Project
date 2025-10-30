import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
import re
import joblib
from itertools import combinations
import os

# -------------------- Helper Functions --------------------

def clean_text(text):
    """Clean text by removing special chars, digits, and lowercasing."""
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces/newlines
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters
    return text.lower().strip()

def extract_text_from_pdf(file):
    """Extract text from an uploaded PDF."""
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def jaccard_similarity(text1, text2):
    """Compute Jaccard similarity between two texts."""
    set1, set2 = set(text1.split()), set(text2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

# -------------------- Streamlit UI --------------------

st.set_page_config(page_title="Jaccard Similarity NLP", page_icon="📄", layout="centered")

st.title("📄 NLP Project: Jaccard Similarity Between PDFs")
st.write("Upload **4 PDF files** to check similarity between them using Jaccard Similarity.")

uploaded_files = st.file_uploader("Upload 4 PDF Files", type=["pdf"], accept_multiple_files=True)

if len(uploaded_files) == 4:
    st.success("✅ All 4 PDFs uploaded successfully!")

    # Extract and clean text
    texts = []
    for file in uploaded_files:
        with st.spinner(f"Extracting text from {file.name}..."):
            extracted = extract_text_from_pdf(file)
            cleaned = clean_text(extracted)
            texts.append(cleaned)

    # Compute Jaccard similarities
    similarity_matrix = [[0 for _ in range(4)] for _ in range(4)]
    for i in range(4):
        similarity_matrix[i][i] = 1.0

    for (i, j) in combinations(range(4), 2):
        sim = jaccard_similarity(texts[i], texts[j])
        similarity_matrix[i][j] = sim
        similarity_matrix[j][i] = sim

    # Create dataframe
    df = pd.DataFrame(similarity_matrix,
                      index=[f"PDF {i+1}" for i in range(4)],
                      columns=[f"PDF {i+1}" for i in range(4)])

    st.subheader("📊 Similarity Matrix")
    st.dataframe(df.style.format("{:.3f}"))

    # Plot heatmap
    st.subheader("🔥 Heatmap of Jaccard Similarity")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

    # Save result with joblib
    joblib.dump(df, "Jaccard_Similarity_Results.pkl")
    st.success("✅ Results saved as `Jaccard_Similarity_Results.pkl`")

    

st.caption("Developed by Sehrish Tariq 💻")


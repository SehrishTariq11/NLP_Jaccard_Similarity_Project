import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
import re
from itertools import combinations

# -------------------- Helper Functions --------------------

def clean_text(text):
    """Clean text by removing special chars, digits, and lowercasing."""
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces/newlines
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters
    return text.lower().strip()

def extract_text_from_pdf(file):
    """Extract text from an uploaded PDF (all pages combined)."""
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

# -------------------- Streamlit App --------------------

st.set_page_config(page_title="Jaccard Similarity NLP", page_icon="📄", layout="centered")

st.title(" NLP Project: Jaccard Similarity Between PDFs")
st.write("Upload **4 PDF files** below to check their similarity using the Jaccard Similarity metric.")

uploaded_files = st.file_uploader("Upload 4 PDF Files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) == 4:
    st.success(" All 4 PDFs uploaded successfully!")

    # Extract and clean text from all PDFs
    texts = []
    for file in uploaded_files:
        with st.spinner(f"Extracting and cleaning text from {file.name}..."):
            extracted = extract_text_from_pdf(file)
            cleaned = clean_text(extracted)
            texts.append(cleaned)

    # Compute pairwise similarities
    similarity_matrix = [[0 for _ in range(4)] for _ in range(4)]
    for i in range(4):
        similarity_matrix[i][i] = 1.0  # Similarity with itself is always 1

    for (i, j) in combinations(range(4), 2):
        sim = jaccard_similarity(texts[i], texts[j])
        similarity_matrix[i][j] = sim
        similarity_matrix[j][i] = sim
        st.write(f"**Similarity between PDF {i+1} and PDF {j+1}: {sim:.4f}**")

    # Create DataFrame
    df = pd.DataFrame(
        similarity_matrix,
        index=[f"PDF {i+1}" for i in range(4)],
        columns=[f"PDF {i+1}" for i in range(4)]
    )

    # Show results
    st.subheader("📊 Similarity Matrix")
    st.dataframe(df.style.format("{:.3f}"))

    # Plot heatmap
    st.subheader(" Heatmap of Jaccard Similarity")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

else:
    st.info("Please upload **exactly 4 PDF files** to start the similarity check.")

st.caption("Developed by Sehrish Tariq 💻")

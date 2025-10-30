import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
import re
from itertools import combinations

# -------------------- Helper Functions --------------------

def clean_text(text):
    """Clean text consistently across environments."""
    # Normalize encoding
    text = text.encode("utf-8", "ignore").decode()
    # Lowercase
    text = text.lower()
    # Keep only alphabets and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(file):
    """Extract text from all pages of a PDF."""
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        try:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        except:
            continue
    return text

def jaccard_similarity(text1, text2):
    """Compute Jaccard similarity between two cleaned texts."""
    words1 = text1.split()
    words2 = text2.split()
    set1, set2 = set(words1), set(words2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return round(intersection / union, 6) if union != 0 else 0

# -------------------- Streamlit App --------------------

st.set_page_config(page_title="Jaccard Similarity NLP", page_icon="📄", layout="centered")

st.title("NLP Project: Jaccard Similarity Between PDFs")
st.write("Upload **4 PDF files** to check their similarity using the Jaccard Similarity metric.")

uploaded_files = st.file_uploader("Upload 4 PDF Files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) == 4:
    st.success("All 4 PDFs uploaded successfully!")

    # Extract and clean text
    texts = []
    for file in uploaded_files:
        with st.spinner(f"Extracting and cleaning text from {file.name}..."):
            extracted = extract_text_from_pdf(file)
            cleaned = clean_text(extracted)
            texts.append(cleaned)

    # Compute pairwise Jaccard similarities
    similarity_matrix = [[0 for _ in range(4)] for _ in range(4)]
    for i in range(4):
        similarity_matrix[i][i] = 1.0  # Same file = 1

    for (i, j) in combinations(range(4), 2):
        sim = jaccard_similarity(texts[i], texts[j])
        similarity_matrix[i][j] = sim
        similarity_matrix[j][i] = sim
        st.write(f"**Similarity between PDF {i+1} and PDF {j+1}: {sim:.6f}**")

    # Create DataFrame
    df = pd.DataFrame(
        similarity_matrix,
        index=[f"PDF {i+1}" for i in range(4)],
        columns=[f"PDF {i+1}" for i in range(4)]
    )

    # Round to 3 decimals for display
    df_rounded = df.round(3)

    # Display table
    st.subheader("Similarity Matrix (Rounded to 3 Decimals)")
    st.table(df_rounded)

    # Plot heatmap
    st.subheader("Heatmap of Jaccard Similarity")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

else:
    st.info("Please upload **exactly 4 PDF files** to start the similarity check.")

st.caption("Developed by Sehrish Tariq 💻")


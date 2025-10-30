import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pdfplumber
import re
from itertools import combinations

# -------------------- Helper Functions --------------------

def clean_text(text):
    """Clean text consistently across systems."""
    text = text.encode("utf-8", "ignore").decode()  # remove invalid chars
    text = text.lower()                             # lowercase
    text = re.sub(r'[^a-z\s]', ' ', text)           # keep only letters & spaces
    text = re.sub(r'\s+', ' ', text).strip()        # normalize spaces
    return text

def extract_text_from_pdf(file):
    """Extract text using pdfplumber for consistency across OS."""
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text

def jaccard_similarity(text1, text2):
    """Compute Jaccard similarity between two cleaned texts."""
    words1, words2 = text1.split(), text2.split()
    set1, set2 = set(words1), set(words2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return round(intersection / union, 6) if union != 0 else 0

# -------------------- Streamlit App --------------------

st.set_page_config(page_title="Jaccard Similarity NLP", page_icon="📄", layout="centered")

st.title("📄 NLP Project: Jaccard Similarity Between PDFs")
st.write("Upload **4 PDF files** to check their similarity using the Jaccard Similarity metric.")

uploaded_files = st.file_uploader("📂 Upload 4 PDF Files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) == 4:
    st.success("✅ All 4 PDFs uploaded successfully!")

    texts = []
    for file in uploaded_files:
        with st.spinner(f"Extracting text from {file.name}..."):
            extracted = extract_text_from_pdf(file)
            cleaned = clean_text(extracted)
            texts.append(cleaned)

    # Show word count (optional check)
    st.write("### 📄 Word Counts per PDF")
    for i, t in enumerate(texts):
        st.write(f"PDF {i+1}: {len(t.split())} words")

    # Compute pairwise Jaccard similarities
    similarity_matrix = [[0 for _ in range(4)] for _ in range(4)]
    for i in range(4):
        similarity_matrix[i][i] = 1.0

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

    # Round for display
    df_rounded = df.round(3)

    # Show results
    st.subheader("📊 Similarity Matrix (Jaccard Index)")
    st.table(df_rounded)

    # Heatmap
    st.subheader("🔥 Heatmap of Jaccard Similarity")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

    # Download option
    csv = df_rounded.to_csv().encode('utf-8')
    st.download_button(
        label="⬇️ Download Similarity Matrix as CSV",
        data=csv,
        file_name='Jaccard_Similarity_Results.csv',
        mime='text/csv'
    )

else:
    st.info("Please upload **exactly 4 PDF files** to start the similarity check.")

st.caption("Developed by Sehrish Tariq 💻")

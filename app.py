import streamlit as st
import requests
import pandas as pd
import numpy as np
from Bio import SeqIO
from io import StringIO

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="AntiMicrobial Peptide Predictor", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #004a99; color: white; }
    </style>
    """, unsafe_allow_html=True)

# 2. API CONFIGURATION
API_URL = "https://router.huggingface.co/models/Pharmson/temp-pharmson-weights-beta"
HF_TOKEN = st.secrets["HF_TOKEN"]

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_api(text):
    # ProtBERT models need spaces between amino acids
    spaced_text = " ".join(list(text.strip().replace(" ", "")))
    response = requests.post(API_URL, headers=headers, json={"inputs": spaced_text})
    return response.json()

# 3. CORE LOGIC
def run_prediction(sequences, names):
    results = []
    progress_bar = st.progress(0)
    
    for i, (seq, name) in enumerate(zip(sequences, names)):
        try:
            output = query_api(seq)
            # The API returns: [[{'label': 'LABEL_0', 'score': 0.1}, {'label': 'LABEL_1', 'score': 0.9}]]
            scores = output[0] 
            
            # Assuming Index 1 is AMP and Index 0 is Non-AMP
            is_amp = scores[1]['score'] > scores[0]['score']
            confidence = scores[1]['score'] if is_amp else scores[0]['score']
            
            results.append({
                "ID": name,
                "Sequence": seq,
                "Prediction": "AMP" if is_amp else "Non-AMP",
                "Confidence": round(float(confidence), 4)
            })
        except Exception:
            results.append({"ID": name, "Sequence": seq, "Prediction": "API Loading/Error", "Confidence": 0.0})
            
        progress_bar.progress((i + 1) / len(sequences))
    return pd.DataFrame(results)

# 4. INTERFACE
st.title("ASAMP")
st.subheader("AntiMicrobial Peptide Predictor")

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    user_seq = st.text_input("Enter Sequence:", placeholder="e.g., KLLKLLK")
    if st.button("Predict"):
        if len(user_seq) < 5:
            st.warning("Sequence must be at least 5 amino acids long.")
        else:
            with st.spinner("Analyzing..."):
                data = query_api(user_seq)
                try:
                    scores = data[0]
                    amp_prob = scores[1]['score']
                    non_amp_prob = scores[0]['score']
                    
                    if amp_prob > non_amp_prob:
                        st.success(f"**Result: AMP** (Confidence: {amp_prob:.2%})")
                    else:
                        st.error(f"**Result: Non-AMP** (Confidence: {non_amp_prob:.2%})")
                except:
                    st.error("Model is still waking up on Hugging Face. Please try again in 30 seconds.")

with tab2:
    st.subheader("Multiple Sequence Input")
    st.write("Paste multiple sequences in FASTA format or upload a file.")
    
    input_type = st.radio("Choose Input Method:", ["Paste FASTA", "Upload .fasta File"])
    
    sequences, names = [], []
    
    if input_type == "Paste FASTA":
        pasted = st.text_area("Paste FASTA here:", height=200, placeholder=">Seq1\nKLLKLLK\n>Seq2\nMAGGG")
        if pasted:
            fasta_io = StringIO(pasted)
            for record in SeqIO.parse(fasta_io, "fasta"):
                sequences.append(str(record.seq))
                names.append(record.id)
    else:
        file = st.file_uploader("Upload File", type=["fasta", "faa", "txt"])
        if file:
            fasta_io = StringIO(file.getvalue().decode("utf-8"))
            for record in SeqIO.parse(fasta_io, "fasta"):
                sequences.append(str(record.seq))
                names.append(record.id)

    if st.button("Analyze All"):
        if not sequences:
            st.error("No sequences found. Ensure they are in FASTA format.")
        else:
            df = run_prediction(sequences, names)
            st.dataframe(df, use_container_width=True)
            st.download_button("Download Results", df.to_csv(index=False), "results.csv", "text/csv")

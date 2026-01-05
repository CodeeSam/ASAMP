import streamlit as st
import os
import zipfile
from transformers import TFBertForSequenceClassification, BertTokenizer
from Bio import SeqIO
from io import StringIO
import numpy as np
import pandas as pd

# ----------------------
# 1. PAGE CONFIGURATION
# ----------------------
st.set_page_config(page_title="AntiMicrobial Peptide Predictor", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #004a99; color: white; }
    </style>
    """, unsafe_allow_html=True)

# ----------------------
# 2. GOOGLE DRIVE MODEL CONFIG
# ----------------------
MODEL_DIR = "model_files"
MODEL_ZIP = "trained_model.zip"
FILE_ID = "1f27bgt-1gJ3iVJWrXnOkYJ6uKPPJLcww"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

import gdown

@st.cache_resource
def load_model():
    # Download ZIP if not exists
    if not os.path.exists(MODEL_DIR):
        st.info("Downloading model... This may take a few minutes ⏳")
        gdown.download(DOWNLOAD_URL, MODEL_ZIP, quiet=False)

        # Extract ZIP
        with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
            zip_ref.extractall(MODEL_DIR)
    
    # Load model and tokenizer
    model = TFBertForSequenceClassification.from_pretrained(MODEL_DIR, from_tf=True)
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    
    return model, tokenizer

model, tokenizer = load_model()

# ----------------------
# 3. CORE LOGIC FOR PREDICTION
# ----------------------
def predict_sequence(seq):
    # Tokenize
    tokens = tokenizer(" ".join(list(seq.strip().replace(" ", ""))),
                       return_tensors="tf", padding=True, truncation=True)
    outputs = model(tokens)
    logits = outputs.logits.numpy()[0]

    # Assuming Index 1 = AMP, Index 0 = Non-AMP
    is_amp = logits[1] > logits[0]
    confidence = float(logits[1]) if is_amp else float(logits[0])
    return "AMP" if is_amp else "Non-AMP", round(confidence, 4)

def run_prediction(sequences, names):
    results = []
    progress_bar = st.progress(0)

    for i, (seq, name) in enumerate(zip(sequences, names)):
        try:
            pred, conf = predict_sequence(seq)
            results.append({
                "ID": name,
                "Sequence": seq,
                "Prediction": pred,
                "Confidence": conf
            })
        except Exception:
            results.append({"ID": name, "Sequence": seq, "Prediction": "Error", "Confidence": 0.0})
        
        progress_bar.progress((i + 1) / len(sequences))
    
    return pd.DataFrame(results)

# ----------------------
# 4. STREAMLIT INTERFACE
# ----------------------
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
                pred, conf = predict_sequence(user_seq)
                if pred == "AMP":
                    st.success(f"**Result: AMP** (Confidence: {conf:.2%})")
                else:
                    st.error(f"**Result: Non-AMP** (Confidence: {conf:.2%})")

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

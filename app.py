import streamlit as st
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification 
import numpy as np
import pandas as pd
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

# 2. MODEL LOADING
HF_MODEL_PATH = "Pharmson/temp-pharmson-weights-beta"

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained(
        "Rostlab/prot_bert", 
        do_lower_case=False,
        local_files_only=False
    )
    
    model = TFBertForSequenceClassification.from_pretrained(
        HF_MODEL_PATH,
        from_pt=False
    )
    
    return tokenizer, model


try:
    with st.spinner("Downloading Model... This may take a few minutes"):
        tokenizer, model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model. Error: {e}")
    st.stop()


# BASE_PROTBERT = "Rostlab/prot_bert"

# @st.cache_resource
# def load_model():
#     tokenizer = BertTokenizer.from_pretrained(BASE_PROTBERT, do_lower_case=False)
#     model = TFBertForSequenceClassification.from_pretrained(HF_MODEL_PATH)
#     return tokenizer, model

# tokenizer, model = load_model()

# 3. CORE LOGIC
def preprocess_sequence(seq):
    return " ".join(list(seq.strip().replace(" ", "")))

def run_prediction(sequences, names):
    results = []
    progress_bar = st.progress(0)
    
    for i, (seq, name) in enumerate(zip(sequences, names)):
        if len(seq.strip()) < 5:
            results.append({"ID": name, "Sequence": seq, "Prediction": "Too Short (<5)", "Confidence": 0.0})
        else:
            processed = preprocess_sequence(seq)
            inputs = tokenizer(processed, return_tensors="tf", padding=True, truncation=True, max_length=190)
            logits = model(inputs).logits
            probs = tf.nn.softmax(logits, axis=1).numpy()[0]
            
            is_amp = np.argmax(probs) == 1
            results.append({
                "ID": name,
                "Sequence": seq,
                "Prediction": "AMP" if is_amp else "Non-AMP",
                "Confidence": round(float(probs[1] if is_amp else probs[0]), 4)
            })
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
            processed = preprocess_sequence(user_seq)
            inputs = tokenizer(processed, return_tensors="tf", padding=True, truncation=True, max_length=190)
            probs = tf.nn.softmax(model(inputs).logits, axis=1).numpy()[0]
            if np.argmax(probs) == 1:
                st.success(f"**Result: AMP** (Confidence: {probs[1]:.2%})")
            else:
                st.error(f"**Result: Non-AMP** (Confidence: {probs[0]:.2%})")

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
            st.error("No sequences found. Ensure they are in FASTA format (starting with '>')")
        else:
            df = run_prediction(sequences, names)
            st.dataframe(df, width="stretch")
            st.download_button("Download Results", df.to_csv(index=False), "results.csv", "text/csv")
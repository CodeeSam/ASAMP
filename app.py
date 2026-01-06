import streamlit as st
import os
import zipfile
import gdown
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from Bio import SeqIO
from io import StringIO
import requests

# ----------------------
# 1. PAGE CONFIGURATION
# ----------------------
st.set_page_config(page_title="ASAMP", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #004a99; color: white; }
    </style>
""", unsafe_allow_html=True)

# ----------------------
# 2. MODEL CONFIG
# ----------------------
MODEL_DIR = "model_files"
MODEL_ZIP = "trained_model.zip"
FILE_ID = "1uNwLrxqTiqI0z0DddrgsZcxUf9EqHx-n"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# ----------------------
# 3. LOAD MODEL (PyTorch)
# ----------------------
@st.cache_resource(show_spinner="Downloading and loading model...")
def load_model():
    # Download ZIP if not exists
    if not os.path.exists(MODEL_DIR):
        st.info("Downloading model... This may take a few minutes ⏳")
        gdown.download(DOWNLOAD_URL, MODEL_ZIP, quiet=False)

        # Extract ZIP
        with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
            zip_ref.extractall(".")

    # Ensure folder exists
    if not os.path.exists(MODEL_DIR):
        st.error("Model folder not found after extraction!")
        st.stop()

    # Load tokenizer & PyTorch model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, device_map="cpu")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# ----------------------
# 4. PREDICTION FUNCTIONS
# ----------------------
def predict_sequence(seq):
    """Predict single sequence."""
    tokens = tokenizer(" ".join(list(seq.strip().replace(" ", ""))),
                       return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**tokens)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    # Index 1 = AMP, 0 = Non-AMP
    if probs[1] > probs[0]:
        return "AMP", float(probs[1])
    else:
        return "Non-AMP", float(probs[0])

# Hugging Face API fallback
API_URL = "https://api-inference.huggingface.co/models/Pharmson/temp-pharmson-weights-beta"
HF_TOKEN = st.secrets.get("HF_TOKEN", "")
headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def query_api(seq):
    spaced_text = " ".join(list(seq.strip().replace(" ", "")))
    response = requests.post(API_URL, headers=headers, json={"inputs": spaced_text})
    return response.json()

def run_prediction(sequences, names):
    """Batch prediction with progress bar."""
    results = []
    progress_bar = st.progress(0)

    for i, (seq, name) in enumerate(zip(sequences, names)):
        try:
            pred, conf = predict_sequence(seq)
            results.append({"ID": name, "Sequence": seq, "Prediction": pred, "Confidence": round(conf, 4)})
        except Exception:
            # fallback to HF API if available
            if headers:
                try:
                    output = query_api(seq)
                    scores = output[0]
                    is_amp = scores[1]['score'] > scores[0]['score']
                    confidence = scores[1]['score'] if is_amp else scores[0]['score']
                    results.append({
                        "ID": name,
                        "Sequence": seq,
                        "Prediction": "AMP" if is_amp else "Non-AMP",
                        "Confidence": round(float(confidence), 4)
                    })
                except Exception:
                    results.append({"ID": name, "Sequence": seq, "Prediction": "API Error", "Confidence": 0.0})
            else:
                results.append({"ID": name, "Sequence": seq, "Prediction": "Error", "Confidence": 0.0})

        progress_bar.progress((i + 1) / len(sequences))
    return pd.DataFrame(results)

# ----------------------
# 5. STREAMLIT INTERFACE
# ----------------------
st.title("ASAMP")
st.subheader("AntiMicrobial Peptide Predictor")

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

# ---- Single Prediction ----
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

# ---- Batch Prediction ----
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

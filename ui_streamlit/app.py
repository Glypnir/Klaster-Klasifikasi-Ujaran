import streamlit as st
import joblib
import torch
import numpy as np
from transformers import BertTokenizer, BertModel

# =======================
# Informasi Aplikasi
# =======================
st.set_page_config(page_title="Klasifikasi Ujaran Kasar", layout="centered")
st.title("🗣️ Klasifikasi & Klasterisasi Ujaran Kasar Bahasa Indonesia")
st.write("""
Aplikasi ini memproses kalimat masukan Anda dan memprediksi:
- Klaster (dengan KMeans)
- Label klasifikasi (dengan SVM)

Model menggunakan IndoBERT sebagai representasi bahasa.
""")

# =======================
# Load Model & Tokenizer
# =======================
@st.cache_resource
def load_all():
    kmeans = joblib.load("ui_streamlit/kmeans_model.pkl")
    svm = joblib.load("svm_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    tokenizer = BertTokenizer.from_pretrained("tokenizer")
    bert = BertModel.from_pretrained("indobenchmark/indobert-base-p1")
    bert.eval()
    bert.to(torch.device("cpu"))
    return kmeans, svm, label_encoder, tokenizer, bert

try:
    kmeans, svm, le, tokenizer, bert_model = load_all()
except Exception as e:
    st.error(f"Gagal memuat model atau tokenizer: {e}")
    st.stop()

# =======================
# Fungsi Embedding
# =======================
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # Ambil token [CLS]
    return embeddings

# =======================
# Input Pengguna
# =======================
input_text = st.text_area("📝 Masukkan kalimat:", height=120)

if st.button("🔍 Prediksi"):
    if input_text.strip() == "":
        st.warning("Kalimat tidak boleh kosong!")
    else:
        with st.spinner("Memproses kalimat..."):
            try:
                embedding = embed_text(input_text)
                cluster = kmeans.predict(embedding)[0]
                label_encoded = svm.predict(embedding)[0]
                label = le.inverse_transform([label_encoded])[0]
            except Exception as e:
                st.error(f"Gagal memproses input: {e}")
                st.stop()

        st.success("✅ Prediksi selesai!")
        st.markdown(f"**Klaster KMeans:** :blue[`{cluster}`]")
        st.markdown(f"**Label SVM:** :orange[`{label}`]")

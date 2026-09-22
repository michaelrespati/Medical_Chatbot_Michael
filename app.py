import re
import random
import warnings
import pandas as pd
import numpy as np
import streamlit as st
from datasets import load_dataset

warnings.filterwarnings("ignore")

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="MedBot — Asisten Kesehatan", page_icon="🏥", layout="centered")

@st.cache_resource
def setup_nltk():
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    return True

setup_nltk()

@st.cache_resource
def load_sbert():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return model, True
    except Exception:
        return None, False

with st.spinner("Menyiapkan model NLP (pertama kali membutuhkan beberapa saat)..."):
    SBERT_MODEL, USE_SBERT = load_sbert()

lemmatizer = WordNetLemmatizer()

indonesian_stopwords = {
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk',
    'pada', 'adalah', 'atau', 'juga', 'dalam', 'tidak', 'akan', 'ada',
    'saya', 'kamu', 'anda', 'ia', 'mereka', 'kami', 'kita', 'bisa',
    'sudah', 'bila', 'jika', 'maka', 'oleh', 'karena', 'apa',
    'bagaimana', 'berapa', 'kapan', 'dimana', 'siapa', 'apakah', 'cara',
    'lebih', 'sangat', 'dapat', 'nya', 'pun', 'lagi', 'belum',
    'telah', 'namun', 'tapi', 'serta', 'meski', 'agar', 'supaya', 'hal',
    'the', 'is', 'are', 'was', 'what', 'how', 'why', 'when', 'where'
}
english_stopwords = set(stopwords.words('english'))
all_stopwords = indonesian_stopwords | english_stopwords

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in all_stopwords and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

@st.cache_data
def load_medical_dataset():
    dataset = load_dataset(
        "UCSD26/medical_dialog", 
        "processed.en", 
        split="train[:3000]", 
        trust_remote_code=True
    )
    
    qa_list = []
    for item in dataset:
        question = item.get('description', '')
        
        answer = ""
        utterances = item.get('utterances', [])
        for utt in utterances:
            if utt.lower().startswith('doctor:'):
                answer = utt.replace('doctor:', '').strip()
                break
        
        if not answer and len(utterances) > 0:
            answer = utterances[-1]
            
        if question and answer:
            qa_list.append({
                "category": "Medical QA",
                "question": question,
                "answer": answer
            })
            
    df_data = pd.DataFrame(qa_list)
    df_data['processed_question'] = df_data['question'].apply(preprocess_text)
    df_data = df_data[df_data['processed_question'].str.strip() != ""].reset_index(drop=True)
    return df_data

with st.spinner("Memuat dataset dari Hugging Face (UCSD26/medical_dialog)..."):
    df = load_medical_dataset()

class MedicalChatbotEngineV3:
    def __init__(self, dataframe, sbert_model, use_sbert, threshold=0.35, top_k=3):
        self.df = dataframe
        self.sbert_model = sbert_model
        self.use_sbert = use_sbert
        self.threshold = threshold if use_sbert else 0.15
        self.top_k = top_k
        self._build_index()
        self._define_rules()

    def _build_index(self):
        if self.use_sbert:
            self.sbert_embeddings = self.sbert_model.encode(
                self.df['question'].tolist(), convert_to_tensor=True
            )
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['processed_question'])

    def _define_rules(self):
        self.rules = {
            'emergency': {
                'patterns': [r'(sesak.*berat|nyeri dada.*berat|tidak.*bernapas|pingsan)', r'(heart attack|cant breathe|chest pain)'],
                'responses': ["🚨 DARURAT! Hubungi 119 atau segera ke IGD!"]
            },
            'greeting': {
                'patterns': [r'\b(halo|hai|hi|hello)\b'],
                'responses': ["👋 Halo! Ada yang bisa saya bantu terkait masalah kesehatan Anda?"]
            }
        }

    def _check_rules(self, text):
        for intent, data in self.rules.items():
            for pattern in data['patterns']:
                if re.search(pattern, text.lower()):
                    return random.choice(data['responses'])
        return None

    def _search_sbert(self, query):
        from sentence_transformers import util
        emb = self.sbert_model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(emb, self.sbert_embeddings)[0].cpu().numpy()
        top_results = np.argsort(-scores)[:self.top_k]
        return [(idx, float(scores[idx])) for idx in top_results]

    def _search_tfidf(self, query):
        processed = preprocess_text(query)
        vec = self.vectorizer.transform([processed])
        scores = cosine_similarity(vec, self.tfidf_matrix).flatten()
        top_results = np.argsort(scores)[::-1][:self.top_k]
        return [(idx, scores[idx]) for idx in top_results]

    def get_response(self, user_input, history):
        if not user_input.strip():
            return "Silakan ketik pertanyaan."

        rule = self._check_rules(user_input)
        if rule:
            return rule

        query = (history[-1] + " " + user_input) if history else user_input

        if self.use_sbert:
            results = self._search_sbert(query)
            method = "SBERT"
        else:
            results = self._search_tfidf(query)
            method = "TF-IDF"

        best_idx, best_score = results[0]
        if best_score < self.threshold:
            return "🤔 Tidak menemukan jawaban yang cukup relevan. Coba jelaskan keluhan Anda dengan kata/bahasa Inggris lain."

        boosted = []
        for idx, score in results:
            text = self.df.iloc[idx]['question']
            bonus = sum(1 for word in user_input.split() if word in text)
            boosted.append((idx, score + 0.05 * bonus))
        best_idx = sorted(boosted, key=lambda x: x[1], reverse=True)[0][0]
        row = self.df.iloc[best_idx]

        return f"**[Match: {best_score:.2f} | Method: {method}]**\n\n{row['answer']}\n\n---\n⚠️ Untuk kondisi serius, selalu konsultasikan ke dokter."

@st.cache_resource
def get_bot(_df, _sbert_model, _use_sbert):
    return MedicalChatbotEngineV3(_df, _sbert_model, _use_sbert)

bot = get_bot(df, SBERT_MODEL, USE_SBERT)

badge = "🟢 Sentence-BERT" if USE_SBERT else "🔵 TF-IDF"
st.markdown(f"## 🏥 MedBot — Asisten Kesehatan  \n`{badge}`")
st.info(f"📚 Dataset terhubung ke Hugging Face: **{len(df)} percakapan medis**.")
st.warning("⚠️ MedBot hanya untuk edukasi, bukan pengganti dokter. Darurat medis: hubungi **119**.")

chip_labels = ["🤕 Sore Throat", "🩸 High Fever", "❤️ Chest Pain", "🫁 Coughing", "🧠 Headache"]
chip_queries = ["sore throat and immune booster", "high fever and headache", "chest pain", "coughing and fever", "headache and dizzy"]
cols = st.columns(len(chip_labels))
chip_clicked = None
for col, label, q in zip(cols, chip_labels, chip_queries):
    if col.button(label, use_container_width=True):
        chip_clicked = q

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history_raw" not in st.session_state:
    st.session_state.history_raw = []

for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

user_input = st.chat_input("Ketik pertanyaan/keluhan kesehatan Anda...")
final_input = chip_clicked or user_input

if final_input:
    st.session_state.messages.append({"role": "user", "content": final_input})
    with st.chat_message("user"):
        st.markdown(final_input)

    response = bot.get_response(final_input, st.session_state.history_raw)
    st.session_state.history_raw.append(final_input)

    st.session_state.messages.append({"role": "bot", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

if st.button("🗑️ Clear chat"):
    st.session_state.messages = []
    st.session_state.history_raw = []

    st.rerun()

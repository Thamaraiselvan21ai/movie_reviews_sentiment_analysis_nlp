import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="CineSent Pro: AI Studio", page_icon="🎬", layout="wide")

# --- CUSTOM CSS FOR GLASSMORPISM UI ---
st.markdown("""
    <style>
    .stApp { background: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), url('https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80'); background-size: cover; }
    .main-box { background: rgba(255, 255, 255, 0.05); padding: 30px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.1); }
    h1, h2, h3 { color: #FF4B4B !important; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS (Using your notebook paths) ---
@st.cache_resource
def load_assets():
    # Loading models from your specified directory
    mnb = joblib.load("D:/Thamarai/GWC/NLP/Models/MRSA_mnb.pkl")
    bnb = joblib.load("D:/Thamarai/GWC/NLP/Models/MRSA_bnb.pkl")
    vectorizer = joblib.load("D:/Thamarai/GWC/NLP/Models/vectorizer.pkl")
    return mnb, bnb, vectorizer

mnb, bnb, vectorizer = load_assets()

def clean_review(text):
    text = re.sub(r'<.*?>', '', text) #
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower() #
    stop_words = set(stopwords.words('english'))
    ss = SnowballStemmer('english')
    return " ".join([ss.stem(w) for w in text.split() if w not in stop_words]) #

# --- UI LAYOUT ---
st.title("🎬 CineSent Pro: AI Narrative Studio")
st.markdown("---")

col_in, col_out = st.columns([1, 1.2], gap="large")

with col_in:
    st.subheader("🖋️ Input Review")
    review_input = st.text_area("What's the verdict on the film?", height=250, placeholder="Start typing the review here...")
    
    analysis_mode = st.radio("Intelligence Mode", ["Standard (MNB)", "Consensus (MNB + BNB)"], horizontal=True)
    analyze_btn = st.button("🚀 RUN AI ANALYTICS")

with col_out:
    if analyze_btn and review_input:
        with st.spinner("Decoding cinematic emotions..."):
            # 1. Prediction logic
            cleaned = clean_review(review_input)
            vect = vectorizer.transform([cleaned]).toarray()
            
            prob_mnb = mnb.predict_proba(vect)[0][1]
            prob_bnb = bnb.predict_proba(vect)[0][1]
            
            final_score = (prob_mnb + prob_bnb) / 2 if "Consensus" in analysis_mode else prob_mnb
            sentiment_label = "POSITIVE" if final_score > 0.5 else "NEGATIVE"
            
            # 2. Innovative Visual: Gauge Meter
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = final_score * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Sentiment: {sentiment_label}", 'font': {'size': 24, 'color': "white"}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#FF4B4B"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(255, 0, 0, 0.3)'},
                        {'range': [40, 60], 'color': 'rgba(255, 255, 0, 0.3)'},
                        {'range': [60, 100], 'color': 'rgba(0, 255, 0, 0.3)'}],
                }
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white", 'family': "Arial"})
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👈 Enter a movie review and hit 'Run' to see the magic.")

# --- BOTTOM SECTION: WORD CLOUD & KEYWORDS ---
if analyze_btn and review_input:
    st.markdown("---")
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("☁️ Narrative Word Cloud")
        wc = WordCloud(background_color="black", colormap="Reds", width=800, height=400).generate(cleaned)
        st.image(wc.to_array(), use_container_width=True)
        
    with c2:
        st.subheader("🔍 Feature Extraction")
        words = cleaned.split()
        unique_words = pd.Series(words).value_counts().head(10)
        st.table(pd.DataFrame({'Stemmed Word': unique_words.index, 'Frequency': unique_words.values}))
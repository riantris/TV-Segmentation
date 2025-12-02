import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="TV Show Segmentation App",
    page_icon="📺",
    layout="centered"
)

# --- LOAD MODEL & SCALER ---
@st.cache_resource
def load_model():
    with open('kmeans_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_model()
except FileNotFoundError:
    st.error("File model tidak ditemukan! Pastikan 'kmeans_model.pkl' dan 'scaler.pkl' ada di folder yang sama.")
    st.stop()

# --- JUDUL & DESKRIPSI ---
st.title("📺 TV Show Content Segmentation")
st.write("""
Aplikasi ini menggunakan **K-Means Clustering** untuk mengelompokkan konten TV 
berdasarkan Popularitas, Rating, dan Jumlah Vote. Masukkan data TV Show baru di bawah ini!
""")

st.markdown("---")

# --- SIDEBAR INPUT ---
st.sidebar.header("Input Fitur TV Show")

def user_input_features():
    popularity = st.sidebar.number_input('Popularity Score (e.g., 1500.5)', min_value=0.0, value=10.0)
    vote_average = st.sidebar.slider('Vote Average (0-10)', 0.0, 10.0, 7.5)
    vote_count = st.sidebar.number_input('Vote Count (e.g., 5000)', min_value=0, value=100)
    
    data = {
        'popularity': popularity,
        'vote_average': vote_average,
        'vote_count': vote_count
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- TAMPILKAN INPUT USER ---
st.subheader("1. Data yang Anda Masukkan")
st.dataframe(input_df)

# --- PREPROCESSING INPUT ---
# User input harus diproses sama persis seperti saat training model
input_transformed = input_df.copy()
input_transformed['popularity_log'] = np.log1p(input_transformed['popularity'])
input_transformed['vote_count_log'] = np.log1p(input_transformed['vote_count'])

# Hapus kolom asli, sisakan yang sudah di-log
input_ready = input_transformed[['popularity_log', 'vote_average', 'vote_count_log']]

# Scaling
input_scaled = scaler.transform(input_ready)

# --- PREDIKSI ---
if st.button('🔍 Prediksi Segmen'):
    cluster_prediction = model.predict(input_scaled)[0]
    
    st.markdown("---")
    st.subheader("2. Hasil Segmentasi")

    # Interpretasi Bisnis (Sesuaikan dengan hasil analisis Cluster 0, 1, 2 Anda)
    # Contoh Interpretasi Hipotetis:
    if cluster_prediction == 1: 
        cluster_name = "🌟 Global Blockbuster"
        desc = "Konten sangat populer, rating tinggi, dan interaksi masif. Pertahankan kualitas!"
        color = "green"
    elif cluster_prediction == 0:
        cluster_name = "💎 Niche / Hidden Gem"
        desc = "Rating bagus tapi popularitas rendah. Perlu strategi marketing lebih agresif."
        color = "blue"
    else:
        cluster_name = "📉 Low Traction / Trash"
        desc = "Rating rendah dan kurang populer. Perlu evaluasi ulang konten."
        color = "red"

    st.markdown(f"### Masuk ke Cluster: :{color}[{cluster_name}]")
    st.info(desc)

    # --- VISUALISASI POSISI ---
    st.subheader("3. Posisi Data di dalam Cluster")
    # Generate dummy background data for visualization context (optional but nice)
    # Disini kita buat visualisasi simpel saja
    
    fig, ax = plt.subplots(figsize=(6, 4))
    # Kita plot posisi berdasarkan Popularitas (Log) vs Rating
    ax.scatter([1, 2, 3], [1, 2, 3], alpha=0) # Dummy empty plot setup
    ax.set_xlabel("Log Popularity (Normalized)")
    ax.set_ylabel("Vote Average (Normalized)")
    ax.set_title("Posisi Input Anda (Titik Merah)")
    
    # Plot titik user (Scaled values)
    # x=Popularity_Log_Scaled, y=Vote_Average_Scaled
    user_x = input_scaled[0][0] 
    user_y = input_scaled[0][1]
    
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.scatter(user_x, user_y, color='red', s=200, label='Your Input', edgecolors='black', zorder=5)
    ax.text(user_x, user_y, f"  {cluster_name}", fontsize=12, color='red', fontweight='bold')
    
    # Add quadrants text (Simplified)
    ax.set_xlim(-3, 5) # Estimasi range dari scaler
    ax.set_ylim(-3, 3) # Estimasi range dari scaler
    
    st.pyplot(fig)
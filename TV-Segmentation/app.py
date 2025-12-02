import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="TV Show Segmentation App",
    page_icon="📺",
    layout="centered"
)

# --- LOAD MODEL & SCALER (VERSI ROBUST/ANTI-GAGAL) ---
@st.cache_resource
def load_model():
    # 1. Dapatkan lokasi absolut folder tempat app.py berada
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Gabungkan path folder dengan nama file
    model_path = os.path.join(current_dir, 'kmeans_model.pkl')
    scaler_path = os.path.join(current_dir, 'scaler.pkl')
    
    # 3. Cek keberadaan file (Debugging)
    if not os.path.exists(model_path):
        st.error(f"❌ ERROR KRITIKAL: File model tidak ditemukan di: {model_path}")
        st.warning("Daftar file yang ada di folder ini:")
        # Tampilkan semua file yang ada di folder server (biar ketahuan salah nama/folder)
        st.code(os.listdir(current_dir))
        return None, None
        
    if not os.path.exists(scaler_path):
        st.error(f"❌ ERROR KRITIKAL: File scaler tidak ditemukan di: {scaler_path}")
        return None, None

    # 4. Load file jika ditemukan
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    return model, scaler

# Panggil fungsi load
model, scaler = load_model()

# Jika load gagal, hentikan aplikasi
if model is None or scaler is None:
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
    # Menggunakan nilai default yang masuk akal
    popularity = st.sidebar.number_input('Popularity Score (e.g., 1500.5)', min_value=0.0, value=15.5)
    vote_average = st.sidebar.slider('Vote Average (0-10)', 0.0, 10.0, 7.5)
    vote_count = st.sidebar.number_input('Vote Count (e.g., 5000)', min_value=0, value=150)
    
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
# Ingat: Kita pakai Log Transform + Scaling di training
input_transformed = input_df.copy()
input_transformed['popularity_log'] = np.log1p(input_transformed['popularity'])
input_transformed['vote_count_log'] = np.log1p(input_transformed['vote_count'])

# Pilih fitur yang sesuai urutan training
input_ready = input_transformed[['popularity_log', 'vote_average', 'vote_count_log']]

# Scaling menggunakan scaler yang sudah diload
input_scaled = scaler.transform(input_ready)

# --- PREDIKSI ---
if st.button('🔍 Prediksi Segmen'):
    cluster_prediction = model.predict(input_scaled)[0]
    
    st.markdown("---")
    st.subheader("2. Hasil Segmentasi")

    # Interpretasi Bisnis
    # Catatan: Label cluster (0, 1, 2) mungkin berbeda tergantung training Anda.
    # Sesuaikan deskripsi ini dengan hasil analisis notebook Anda.
    if cluster_prediction == 1: 
        cluster_name = "🌟 Global Blockbuster"
        desc = "Konten sangat populer dengan interaksi masif. Pertahankan kualitas!"
        color = "green"
    elif cluster_prediction == 0:
        cluster_name = "💎 Niche / Hidden Gem"
        desc = "Rating bagus tapi popularitas belum meledak. Perlu dorongan marketing."
        color = "blue"
    else:
        cluster_name = "📉 Low Traction / Mass Market"
        desc = "Rating rendah atau rata-rata. Perlu evaluasi konten."
        color = "red"

    st.markdown(f"### Masuk ke Cluster: :{color}[{cluster_name}]")
    st.info(desc)

    # --- VISUALISASI POSISI ---
    st.subheader("3. Posisi Data di dalam Cluster")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot Dummy Centroids (Opsional, untuk konteks visual)
    # Karena kita tidak upload data asli (CSV), kita buat visualisasi relatif saja
    ax.set_title("Posisi Input Anda (Relatif terhadap Ruang Fitur)")
    ax.set_xlabel("Popularitas & Interaksi (Scaled)")
    ax.set_ylabel("Kualitas Rating (Scaled)")
    
    # Plot titik user (Scaled values)
    # x = Kombinasi Popularitas (index 0)
    # y = Rating (index 1)
    user_x = input_scaled[0][0] 
    user_y = input_scaled[0][1]
    
    # Gambar grid dan kuadran
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.axvline(0, color='grey', linewidth=0.8)
    
    # Titik User
    ax.scatter(user_x, user_y, color='red', s=250, label='Your Content', edgecolors='black', zorder=5, marker='*')
    
    # Anotasi
    ax.text(user_x + 0.2, user_y, f"  {cluster_name}\n  (You are here)", fontsize=10, color='darkred', fontweight='bold')
    
    # Atur limit agar titik selalu terlihat di tengah
    ax.set_xlim(user_x - 3, user_x + 3)
    ax.set_ylim(user_y - 3, user_y + 3)
    
    st.pyplot(fig)

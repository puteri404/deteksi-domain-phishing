import streamlit as st
import pandas as pd
import joblib
import Levenshtein
import jellyfish

# ==============================================================================
# 1. KONFIGURASI HALAMAN & RESOURCE LOADING
# ==============================================================================
st.set_page_config(
    page_title="Deteksi Typosquatting Phishing",
    page_icon="🛡️",
    layout="centered"
)

# Menggunakan cache agar model tidak di-load ulang setiap ada input baru
@st.cache_resource
def load_artifacts():
    model = joblib.load('best_model.pkl')
    referensi = joblib.load('referensi_domain.pkl')
    return model, referensi

try:
    model, referensi_domain = load_artifacts()
except Exception as e:
    st.error(f"Gagal memuat artefak model. Pastikan 'best_model.pkl' dan 'referensi_domain.pkl' berada di folder yang sama. Error: {e}")
    st.stop()

# ==============================================================================
# 2. FUNGSI EKSTRAKSI FITUR REAL-TIME
# ==============================================================================
def hitung_fitur_domain(domain, referensi_list):
    domain = str(domain).strip().lower()
    
    min_levenshtein = float('inf')
    max_jaro_winkler = 0.0
    domain_paling_mirip = ""
    
    # Mencari domain terdekat dari 1000 referensi Tranco
    for ref in referensi_list:
        lev_dist = Levenshtein.distance(domain, ref)
        if lev_dist < min_levenshtein:
            min_levenshtein = lev_dist
            
        jw_sim = jellyfish.jaro_winkler_similarity(domain, ref)
        if jw_sim > max_jaro_winkler:
            max_jaro_winkler = jw_sim
            domain_paling_mirip = ref
            
        if lev_dist == 0 and jw_sim == 1.0:
            break
            
    panjang_domain = len(domain)
    
    return min_levenshtein, max_jaro_winkler, panjang_domain, domain_paling_mirip

# ==============================================================================
# 3. ANTARMUKA PENGGUNA (UI DESIGN)
# ==============================================================================
st.title("🛡️ Deteksi Website Phishing Berbasis Typosquatting")
st.write("Aplikasi ini mendeteksi apakah suatu domain merupakan indikasi *phishing* berbasis ejaan tiruan (*typosquatting*) menggunakan metrik jarak string dan algoritma Machine Learning.")
st.markdown("---")

# Input dari pengguna
input_domain = st.text_input(
    "Masukkan Nama Domain Mencurigakan:", 
    placeholder="contoh: g00gle.com, klikbca-indonesia.com, facebook.id"
)

if st.button("Analisis Keamanan Domain", type="primary"):
    if input_domain:
        # Bersihkan input jika pengguna memasukkan http:// atau https://
        clean_domain = input_domain.replace("https://", "").replace("http://", "").split("/")[0]
        
        with st.spinner('Sedang menganalisis struktur domain...'):
            # 1. Ekstraksi Fitur
            min_lev, max_jw, panjang, ref_mirip = hitung_fitur_domain(clean_domain, referensi_domain)
            
            # 2. Buat DataFrame Input untuk Model
            input_data = pd.DataFrame([{
                'min_levenshtein': min_lev,
                'max_jaro_winkler': max_jw,
                'panjang_domain': panjang
            }])
            
            # 3. Prediksi Model
            prediksi = model.predict(input_data)[0]
            probabilitas = model.predict_proba(input_data)[0]
            
        # ==============================================================================
        # 4. TAMPILKAN HASIL PREDIKSI
        # ==============================================================================
        st.subheader("Hasil Analisis:")
        
        if prediksi == 1:
            st.error(f"🚨 **PERINGATAN: Domain ini Terindikasi PHISHING / TYPOSQUATTING!**")
            st.metric(label="Tingkat Keyakinan Model (Phishing)", value=f"{probabilitas[1]*100:.2f}%")
        else:
            st.success(f"✅ **AMAN: Domain ini Teridentifikasi NORMAL / SAH.**")
            st.metric(label="Tingkat Keyakinan Model (Aman)", value=f"{probabilitas[0]*100:.2f}%")
            
        # ==============================================================================
        # 5. EXPLAINABLE AI (XAI) - PENJELASAN INDIKATOR FITUR
        # ==============================================================================
        st.markdown("### 📊 Indikator Fitur Jarak String (Analisis Eksploratif):")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Levenshtein Distance", value=int(min_lev), help="Jumlah perubahan karakter minimum untuk menjadi domain populer asli. Makin kecil makin rawan typosquatting.")
        with col2:
            st.metric(label="Jaro-Winkler Similarity", value=f"{max_jw:.4f}", help="Skor kemiripan tekstual (0 hingga 1). Nilai mendekati 1.000 mengindikasikan kemiripan visual yang sangat tinggi.")
        with col3:
            st.metric(label="Panjang Karakter", value=int(panjang))
            
        st.info(f"💡 **Analisis Konteks:** Domain yang Anda masukkan memiliki tingkat kemiripan tertinggi dengan domain populer asli: **`{ref_mirip}`**.")

    else:
        st.warning("Silakan masukkan nama domain terlebih dahulu.")

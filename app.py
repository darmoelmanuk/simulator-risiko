import streamlit as st
import numpy as np
import joblib

# Memuat model dan scaler (Fungsi asli tidak diubah)
model = joblib.load("model_risiko_v1.joblib")
scaler = joblib.load("scaler_risiko_v1.joblib")

# Konfigurasi Halaman
st.set_page_config(
    page_title="Simulator Risiko Mesin",
    page_icon="⚙️",
    layout="centered"
)

# --- BAGIAN HEADER ---
st.title("Simulator Risiko Mesin")
st.markdown("Masukkan kondisi parameter mesin saat ini pada form di bawah untuk memprediksi tingkat risiko secara **real-time**.")
st.divider()

# --- BAGIAN INPUT ---
st.subheader("⚙️ Parameter Operasional")

# Menggunakan kolom agar input berdampingan
col1, col2 = st.columns(2)

with col1:
    suhu = st.number_input(
        "🌡️ Suhu Mesin (°C)",
        value=80.0,
        help="Contoh: 80.0"
    )

with col2:
    getaran = st.number_input(
        "📳 Getaran Mesin (mm/s)",
        value=5.0,
        help="Contoh: 5.0"
    )

# Memberi sedikit ruang sebelum tombol
st.write("") 

# --- BAGIAN PROSES & OUTPUT ---
# Membuat tombol lebih menonjol dengan type="primary"
if st.button("Simulasikan", type="primary", use_container_width=True):
    
    # Logika data dan prediksi (Fungsi asli tidak diubah sama sekali)
    data = np.array([[suhu, getaran]])
    data_scaled = scaler.transform(data)
    hasil = model.predict(data_scaled)
    
    # Menampilkan hasil dengan layout yang rapi
    st.divider()
    st.subheader("Hasil Simulasi")
    
    # Menambahkan elemen st.metric untuk visualisasi angka yang lebih profesional
    st.metric(label="Indikator Risiko", value=f"{hasil[0]:.2f}")
    
    # Output pesan sukses asli
    st.success(
        f"Skor Risiko : {hasil[0]:.2f}"
    )
    
    # Logika peringatan asli
    if suhu > 120 or suhu < 10:
        st.warning(
            "⚠ Input berada di luar data latihan."
        )

st.divider()
# Footer asli
st.caption("Model : model_risiko_v1.joblib")
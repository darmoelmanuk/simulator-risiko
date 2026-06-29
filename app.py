"""
==========================================================================
SIMULATOR RISIKO MESIN - DASHBOARD UAS MACHINE LEARNING
==========================================================================
Aplikasi Streamlit untuk memprediksi risiko mesin menggunakan model
Linear Regression yang telah dilatih dengan StandardScaler.

Menjalankan aplikasi:
    streamlit run app.py
==========================================================================
"""

# ===================================================
# SECTION 1: IMPORT LIBRARY
# ===================================================
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List


# ===================================================
# SECTION 2: KONSTANTA APLIKASI
# ===================================================
MODEL_PATH: str = os.path.join("models", "model_risiko_v1.joblib")
SCALER_PATH: str = os.path.join("models", "scaler_risiko_v1.joblib")

MODEL_NAME: str = "model_risiko_v1.joblib"
APP_VERSION: str = "1.0"
DEVELOPER_NAME: str = "Darmo Wiyono"
MODEL_VERSION: str = "v1"

BATAS_SUHU_ATAS: float = 120.0
BATAS_SUHU_BAWAH: float = 10.0

FEATURE_NAMES: List[str] = ["Suhu Mesin", "Getaran Mesin"]

SAW_BOBOT: List[float] = [0.4, 0.3, 0.3]
SAW_KOLOM: List[str] = ["Risiko Mesin", "Kriteria Kedua", "Kriteria Ketiga"]
SAW_ALTERNATIF: List[str] = ["Alternatif 1 (Input Saat Ini)", "Alternatif 2", "Alternatif 3"]


# ===================================================
# SECTION 3: PAGE CONFIG
# ===================================================
st.set_page_config(
    page_title="Simulator Risiko Mesin",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===================================================
# SECTION 4: CACHE - LOAD MODEL DAN SCALER
# ===================================================
@st.cache_resource
def load_model_dan_scaler(model_path: str, scaler_path: str) -> Tuple[Optional[object], Optional[object], bool]:
    """
    Memuat model dan scaler dari disk menggunakan joblib.
    Mengembalikan tuple (model, scaler, status_berhasil).
    Tidak melempar exception ke pemanggil; semua error ditangani di sini.
    """
    try:
        model_terload = joblib.load(model_path)
        scaler_terload = joblib.load(scaler_path)
        return model_terload, scaler_terload, True
    except Exception:
        return None, None, False


model, scaler, status_load_berhasil = load_model_dan_scaler(MODEL_PATH, SCALER_PATH)
model_loaded: bool = model is not None
scaler_loaded: bool = scaler is not None


# ===================================================
# SECTION 5: FUNGSI BANTUAN (HELPER FUNCTIONS)
# ===================================================
def prediksi_risiko(suhu_mesin: float, getaran_mesin: float) -> Optional[float]:
    """
    Melakukan prediksi risiko menggunakan model dan scaler yang sudah dimuat.
    WAJIB melakukan scaling sebelum prediksi.
    Mengembalikan None apabila terjadi kegagalan.
    """
    try:
        data_input = np.array([[suhu_mesin, getaran_mesin]])
        data_scaled = scaler.transform(data_input)
        hasil = model.predict(data_scaled)
        return float(hasil[0])
    except Exception as error:
        st.error(f"Gagal melakukan prediksi: {error}")
        return None


def cek_drift(suhu_mesin: float) -> Tuple[bool, str]:
    """
    Mendeteksi apakah nilai suhu berada di luar distribusi data latihan.
    Mengembalikan tuple (status_drift, pesan).
    """
    try:
        if suhu_mesin > BATAS_SUHU_ATAS or suhu_mesin < BATAS_SUHU_BAWAH:
            pesan = "⚠ Input berada di luar distribusi data latihan."
            return True, pesan
        else:
            pesan = "Input masih berada dalam distribusi data latihan."
            return False, pesan
    except Exception:
        return False, "Tidak dapat melakukan pengecekan drift."


def hitung_feature_importance_shap(suhu_mesin: float, getaran_mesin: float) -> Tuple[Optional[np.ndarray], str]:
    """
    Menghitung feature importance menggunakan SHAP (LinearExplainer).
    Jika SHAP gagal dijalankan, fallback otomatis ke model.coef_.
    Mengembalikan tuple (array_importance, metode_yang_dipakai).
    """
    try:
        import shap

        data_input = np.array([[suhu_mesin, getaran_mesin]])
        data_scaled = scaler.transform(data_input)

        background_data = np.zeros((1, data_scaled.shape[1]))
        explainer = shap.LinearExplainer(model, background_data)
        shap_values = explainer.shap_values(data_scaled)

        nilai_importance = np.array(shap_values[0])
        return nilai_importance, "SHAP"
    except Exception:
        try:
            nilai_importance = np.array(model.coef_)
            return nilai_importance, "Fallback (model.coef_)"
        except Exception:
            return None, "Gagal Total"


def buat_grafik_feature_importance(nilai_importance: np.ndarray, nama_fitur: List[str]) -> plt.Figure:
    """
    Membuat grafik batang horizontal untuk menampilkan feature importance.
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    warna_batang = ["#1f77b4" if nilai >= 0 else "#d62728" for nilai in nilai_importance]
    ax.barh(nama_fitur, nilai_importance, color=warna_batang)
    ax.set_xlabel("Tingkat Pengaruh")
    ax.set_title("Feature Importance terhadap Risiko Mesin")
    ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    return fig


def buat_penjelasan_ai(nilai_importance: np.ndarray, nama_fitur: List[str]) -> str:
    """
    Membuat penjelasan sederhana berbasis fitur dengan nilai absolut tertinggi.
    """
    try:
        index_tertinggi = int(np.argmax(np.abs(nilai_importance)))
        nama_fitur_dominan = nama_fitur[index_tertinggi]
        arah_pengaruh = "meningkatkan" if nilai_importance[index_tertinggi] >= 0 else "menurunkan"
        penjelasan = f"{nama_fitur_dominan} menjadi faktor utama yang {arah_pengaruh} risiko."
        return penjelasan
    except Exception:
        return "Tidak dapat menghasilkan penjelasan AI saat ini."


def hitung_saw(matriks_keputusan: np.ndarray, bobot_kriteria: List[float]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Menghitung metode SAW (Simple Additive Weighting).
    Mengembalikan DataFrame matriks, normalisasi, dan hasil skor+ranking.
    """
    df_matriks = pd.DataFrame(matriks_keputusan, columns=SAW_KOLOM, index=SAW_ALTERNATIF)

    nilai_maksimum_kolom = matriks_keputusan.max(axis=0)
    matriks_normalisasi = matriks_keputusan / nilai_maksimum_kolom
    df_normalisasi = pd.DataFrame(matriks_normalisasi, columns=SAW_KOLOM, index=SAW_ALTERNATIF)

    skor_akhir = matriks_normalisasi.dot(np.array(bobot_kriteria))
    df_skor = pd.DataFrame({"Skor Akhir": skor_akhir}, index=SAW_ALTERNATIF)
    df_skor["Ranking"] = df_skor["Skor Akhir"].rank(ascending=False).astype(int)
    df_skor = df_skor.sort_values("Ranking")

    return df_matriks, df_normalisasi, df_skor


# ===================================================
# SECTION 6: HEADER
# ===================================================
st.title("⚙️ Simulator Risiko Mesin")
st.write(
    "Dashboard ini digunakan untuk mensimulasikan tingkat risiko mesin "
    "berdasarkan parameter operasional menggunakan model Machine Learning "
    "(Linear Regression) yang telah dilatih sebelumnya."
)
st.divider()


# ===================================================
# SECTION 7: SIDEBAR
# ===================================================
with st.sidebar:
    st.header("ℹ️ Informasi Model")

    st.markdown(f"**Model**\n\nLinear Regression")
    st.markdown(f"**Version**\n\n{MODEL_VERSION}")

    status_text = "Ready" if (model_loaded and scaler_loaded) else "Not Ready"
    st.markdown(f"**Status**\n\n{status_text}")

    st.markdown(f"**Developer**\n\n{DEVELOPER_NAME}")

    st.divider()
    st.subheader("✅ Replayability Checklist")

    if model_loaded:
        st.success("Model Loaded")
    else:
        st.error("Model Loaded")

    if scaler_loaded:
        st.success("Scaler Loaded")
    else:
        st.error("Scaler Loaded")

    st.success("Streamlit Ready")


# ===================================================
# SECTION 8: PERINGATAN JIKA MODEL/SCALER GAGAL DIMUAT
# ===================================================
if not model_loaded or not scaler_loaded:
    st.warning(
        "Model atau scaler gagal dimuat. Pastikan file "
        f"`{MODEL_PATH}` dan `{SCALER_PATH}` tersedia di repository."
    )


# ===================================================
# SECTION 9: INPUT PENGGUNA
# ===================================================
st.subheader("📥 Input Parameter Mesin")

kolom_input_1, kolom_input_2 = st.columns(2)

with kolom_input_1:
    suhu_mesin_input: float = st.number_input(
        label="Suhu Mesin (°C)",
        min_value=0.0,
        max_value=200.0,
        value=75.0,
        step=1.0,
        help="Masukkan suhu mesin saat ini dalam satuan derajat Celsius.",
    )

with kolom_input_2:
    getaran_mesin_input: float = st.number_input(
        label="Getaran Mesin (mm/s)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=0.1,
        help="Masukkan tingkat getaran mesin saat ini dalam satuan mm/s.",
    )

st.divider()


# ===================================================
# SECTION 10: TOMBOL SIMULASI
# ===================================================
tombol_simulasi_ditekan: bool = st.button(
    "Simulasikan",
    type="primary",
    use_container_width=True,
)


# ===================================================
# SECTION 11: PROSES PREDIKSI, DRIFT, SHAP, DAN WHAT-IF
# ===================================================
if tombol_simulasi_ditekan:
    if not model_loaded or not scaler_loaded:
        st.error("Tidak dapat melakukan simulasi karena model atau scaler belum siap.")
    else:
        # ---------------------------------------------
        # 11.1 Prediksi Machine Learning
        # ---------------------------------------------
        st.subheader("🎯 Hasil Prediksi Risiko")

        hasil_prediksi = prediksi_risiko(suhu_mesin_input, getaran_mesin_input)

        if hasil_prediksi is not None:
            kolom_metrik_1, kolom_metrik_2 = st.columns(2)
            with kolom_metrik_1:
                st.metric(label="Skor Risiko Mesin", value=f"{hasil_prediksi:.2f}")
            with kolom_metrik_2:
                st.success("Prediksi berhasil dihitung menggunakan model Linear Regression.")

            st.divider()

            # ---------------------------------------------
            # 11.2 Drift Detection
            # ---------------------------------------------
            st.subheader("📡 Drift Detection")

            status_drift, pesan_drift = cek_drift(suhu_mesin_input)
            if status_drift:
                st.warning(pesan_drift)
            else:
                st.info(pesan_drift)

            st.divider()

            # ---------------------------------------------
            # 11.3 AI Explainability (SHAP / Fallback)
            # ---------------------------------------------
            st.subheader("🧠 AI Explainability")

            nilai_importance, metode_dipakai = hitung_feature_importance_shap(
                suhu_mesin_input, getaran_mesin_input
            )

            if nilai_importance is not None:
                st.caption(f"Metode yang digunakan: {metode_dipakai}")

                kolom_shap_1, kolom_shap_2 = st.columns([2, 1])

                with kolom_shap_1:
                    try:
                        figur_importance = buat_grafik_feature_importance(nilai_importance, FEATURE_NAMES)
                        st.pyplot(figur_importance)
                    except Exception as error:
                        st.warning(f"Grafik feature importance tidak dapat ditampilkan: {error}")

                with kolom_shap_2:
                    penjelasan_ai = buat_penjelasan_ai(nilai_importance, FEATURE_NAMES)
                    st.info(penjelasan_ai)
            else:
                st.warning("Penjelasan AI tidak dapat dihitung saat ini.")

            st.divider()

            # ---------------------------------------------
            # 11.4 What If Analysis
            # ---------------------------------------------
            st.subheader("🔄 What If Analysis")

            try:
                suhu_baseline = 75.0
                getaran_baseline = 5.0

                hasil_baseline = prediksi_risiko(suhu_baseline, getaran_baseline)

                if hasil_baseline is not None:
                    delta_perubahan = hasil_prediksi - hasil_baseline

                    kolom_whatif_1, kolom_whatif_2, kolom_whatif_3 = st.columns(3)
                    with kolom_whatif_1:
                        st.metric(label="Baseline", value=f"{hasil_baseline:.2f}")
                    with kolom_whatif_2:
                        st.metric(label="Prediksi Baru", value=f"{hasil_prediksi:.2f}")
                    with kolom_whatif_3:
                        st.metric(
                            label="Perubahan",
                            value=f"{hasil_prediksi:.2f}",
                            delta=f"{delta_perubahan:.2f}",
                        )
                else:
                    st.warning("Tidak dapat menghitung baseline untuk What If Analysis.")
            except Exception as error:
                st.warning(f"What If Analysis tidak dapat ditampilkan: {error}")

            st.divider()

            # ---------------------------------------------
            # 11.5 SAW (Simple Additive Weighting)
            # ---------------------------------------------
            st.subheader("📊 Simple Additive Weighting (SAW)")

            try:
                matriks_keputusan = np.array([
                    [hasil_prediksi, 80, 90],
                    [55, 70, 75],
                    [40, 90, 95],
                ])

                df_matriks_saw, df_normalisasi_saw, df_skor_saw = hitung_saw(
                    matriks_keputusan, SAW_BOBOT
                )

                st.markdown("**Matriks Keputusan**")
                st.dataframe(df_matriks_saw, use_container_width=True)

                st.markdown("**Matriks Normalisasi**")
                st.dataframe(df_normalisasi_saw, use_container_width=True)

                st.markdown("**Skor dan Ranking**")
                st.dataframe(df_skor_saw, use_container_width=True)

            except Exception as error:
                st.warning(f"Perhitungan SAW tidak dapat ditampilkan: {error}")

        else:
            st.error("Prediksi gagal dilakukan. Periksa kembali input atau model.")

else:
    st.info("Silakan masukkan parameter mesin dan tekan tombol **Simulasikan** untuk memulai.")


# ===================================================
# SECTION 12: FOOTER
# ===================================================
st.divider()

kolom_footer_1, kolom_footer_2 = st.columns(2)
with kolom_footer_1:
    st.caption(f"Model: {MODEL_NAME}")
with kolom_footer_2:
    st.caption(f"Version: {APP_VERSION}")
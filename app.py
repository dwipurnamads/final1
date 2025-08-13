import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ======================
# Load model & metadata
# ======================
@st.cache_resource
def load_bundle():
    try:
        with open('model_deployment/full_pipeline.pkl', 'rb') as f:
            bundle = pickle.load(f)
        model = bundle.get('model')
        scaler = bundle.get('scaler')

        # Cari daftar kolom fitur yang dipakai saat training
        feature_columns = bundle.get('feature_columns')
        if feature_columns is None:
            # fallback: coba dari scaler atau model (harusnya ada kalau fit pakai DataFrame)
            if scaler is not None and hasattr(scaler, 'feature_names_in_'):
                feature_columns = list(scaler.feature_names_in_)
            elif model is not None and hasattr(model, 'feature_names_in_'):
                feature_columns = list(model.feature_names_in_)
            else:
                raise ValueError(
                    "Daftar kolom fitur tidak ditemukan. "
                    "Simpan X_train.columns ke 'feature_columns' saat training."
                )
        return model, scaler, feature_columns
    except FileNotFoundError:
        st.error("File model tidak ditemukan. Pastikan 'model_deployment/full_pipeline.pkl' ada di repo.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        st.stop()

model, scaler, expected_cols = load_bundle()

# ======================
# Konfigurasi fitur
# ======================
numerical_features_for_log = [
    'song_duration_ms', 'acousticness', 'instrumentalness',
    'liveness', 'speechiness', 'tempo'
]
categorical_features = ['audio_mode', 'key', 'time_signature']

# ======================
# UI
# ======================
st.title("Prediksi Popularitas Lagu")
st.markdown("""
Aplikasi ini memprediksi popularitas lagu berdasarkan fitur audio.
Masukkan nilai fitur di bawah ini.
""")

col1, col2, col3 = st.columns(3)
with col1:
    song_duration_ms = st.number_input("Song Duration (ms)", min_value=0.0, value=200000.0)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5, 0.01)
    danceability = st.slider("Danceability", 0.0, 1.0, 0.5, 0.01)
    energy = st.slider("Energy", 0.0, 1.0, 0.5, 0.01)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0, 0.0001)

with col2:
    liveness = st.slider("Liveness", 0.0, 1.0, 0.1, 0.001)
    loudness = st.slider("Loudness (dB)", -60.0, 0.0, -10.0, 0.1)
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05, 0.001)
    tempo = st.number_input("Tempo (bpm)", min_value=0.0, value=120.0)
    audio_valence = st.slider("Audio Valence", 0.0, 1.0, 0.5, 0.01)

with col3:
    audio_mode = st.radio("Audio Mode", options=[0, 1], format_func=lambda x: "Minor" if x == 0 else "Major")
    key = st.selectbox("Key", options=list(range(12)))
    time_signature = st.selectbox("Time Signature", options=[0, 1, 3, 4, 5])

# Konstruksi DataFrame input mentah
input_data = {
    'song_duration_ms': song_duration_ms,
    'acousticness': acousticness,
    'danceability': danceability,
    'energy': energy,
    'instrumentalness': instrumentalness,
    'liveness': liveness,
    'loudness': loudness,
    'speechiness': speechiness,
    'tempo': tempo,
    'audio_valence': audio_valence,
    'audio_mode': audio_mode,
    'key': key,
    'time_signature': time_signature
}
input_df = pd.DataFrame([input_data])

# ======================
# Preprocessing konsisten
# ======================

# 1) Log1p untuk fitur skewed (sesuai training)
for f in numerical_features_for_log:
    if f in input_df.columns:
        # log1p(0) aman = 0; jika negatif jangan dilog
        input_df[f] = input_df[f].apply(lambda x: np.log1p(x) if x >= 0 else x)

# 2) One-hot encoding untuk kategori
input_df_encoded = pd.get_dummies(input_df, columns=categorical_features, drop_first=False)

# 3) Align kolom ke expected_cols (tambahkan yang hilang = 0; buang yang ekstra)
X_aligned = input_df_encoded.reindex(columns=expected_cols, fill_value=0)

# 4) Scaling (jika scaler tersedia)
# Penting: scaler di-fit pada SELURUH matriks fitur (termasuk dummies),
# jadi di sini kita transform seluruh X_aligned, bukan subset numerik saja.
if scaler is not None:
    X_scaled_array = scaler.transform(X_aligned)
    X_ready = pd.DataFrame(X_scaled_array, columns=expected_cols, index=X_aligned.index)
else:
    X_ready = X_aligned

# ======================
# Predict
# ======================
if st.button("Prediksi Popularitas"):
    try:
        pred = model.predict(X_ready)
        st.subheader(f"Prediksi Popularitas Lagu: {pred[0]:.2f}")
    except Exception as e:
        # Debug helper: tampilkan mismatch jika ada
        missing = set(expected_cols) - set(X_ready.columns)
        extra = set(X_ready.columns) - set(expected_cols)
        if missing:
            st.warning(f"Missing cols: {sorted(list(missing))[:10]} ...")
        if extra:
            st.info(f"Extra cols: {sorted(list(extra))[:10]} ...")
        st.error(f"Gagal melakukan prediksi: {e}")

st.markdown("---")
st.caption("Catatan: Jika model dilatih dengan pipeline yang berbeda, pertimbangkan untuk menyimpan seluruh pipeline preprocessing+model.")

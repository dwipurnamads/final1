import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

st.set_page_config(page_title="Prediksi Popularitas Lagu", layout="wide")

# ---------------------------------------------------------
# 1) Load bundle (model, scaler, metadata) dengan fallback
# ---------------------------------------------------------
@st.cache_resource
def load_bundle():
    # coba dua lokasi umum
    candidate_paths = [
        "full_pipeline.pkl",
        "full_pipeline.pkl",
    ]
    last_err = None
    for p in candidate_paths:
        try:
            with open(p, "rb") as f:
                bundle = pickle.load(f)
            return bundle, p
        except Exception as e:
            last_err = e
    raise FileNotFoundError(f"Gagal memuat file pipeline. Coba cek path: {candidate_paths}. Detail: {last_err}")

try:
    bundle, bundle_path = load_bundle()
except Exception as e:
    st.error(f"Tidak bisa memuat model: {e}")
    st.stop()

model  = bundle.get("model", None)
scaler = bundle.get("scaler", None)

# ---------------------------------------------------------
# 2) Tentukan daftar kolom yang DIHARAPKAN MODEL
#    Urutan preferensi:
#    a) bundle['feature_columns'] (disimpan saat training)
#    b) model.feature_names_in_
#    c) (terakhir sekali) scaler.feature_names_in_
# ---------------------------------------------------------
expected_cols = bundle.get("feature_columns", None)
if expected_cols is None and model is not None and hasattr(model, "feature_names_in_"):
    expected_cols = list(model.feature_names_in_)
if expected_cols is None and scaler is not None and hasattr(scaler, "feature_names_in_"):
    # kurang ideal, tapi lebih baik daripada tidak sama sekali
    expected_cols = list(scaler.feature_names_in_)

if expected_cols is None:
    st.error(
        "Tidak menemukan daftar kolom fitur yang diharapkan.\n"
        "Solusi cepat: saat training, simpan X_train.columns ke bundle['feature_columns']."
    )
    st.stop()

# ---------------------------------------------------------
# 3) Konfigurasi input & UI
# ---------------------------------------------------------
# fitur yang kamu log-transform di preprocessing
numerical_features_for_log = [
    "song_duration_ms", "acousticness", "instrumentalness",
    "liveness", "speechiness", "tempo"
]
categorical_features = ["audio_mode", "key", "time_signature"]

st.title("Prediksi Popularitas Lagu")
st.caption(f"Model: {os.path.basename(bundle_path)}")

st.markdown("""
Aplikasi ini memprediksi popularitas lagu berdasarkan fitur-fitur audio.
Masukkan nilai fitur di bawah ini.
""")

col1, col2, col3 = st.columns(3)
with col1:
    song_duration_ms = st.number_input("Song Duration (ms)", min_value=0.0, value=200000.0)
    acousticness     = st.slider("Acousticness", 0.0, 1.0, 0.5, 0.01)
    danceability     = st.slider("Danceability", 0.0, 1.0, 0.5, 0.01)
    energy           = st.slider("Energy", 0.0, 1.0, 0.5, 0.01)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0, 0.0001)
with col2:
    liveness     = st.slider("Liveness", 0.0, 1.0, 0.1, 0.001)
    loudness     = st.slider("Loudness (dB)", -60.0, 0.0, -10.0, 0.1)
    speechiness  = st.slider("Speechiness", 0.0, 1.0, 0.05, 0.001)
    tempo        = st.number_input("Tempo (bpm)", min_value=0.0, value=120.0)
    audio_valence= st.slider("Audio Valence", 0.0, 1.0, 0.5, 0.01)
with col3:
    audio_mode    = st.radio("Audio Mode", options=[0, 1], format_func=lambda x: "Minor" if x == 0 else "Major")
    key           = st.selectbox("Key", options=list(range(12)))
    time_signature= st.selectbox("Time Signature", options=[0, 1, 3, 4, 5])

raw_input = {
    "song_duration_ms": song_duration_ms,
    "acousticness": acousticness,
    "danceability": danceability,
    "energy": energy,
    "instrumentalness": instrumentalness,
    "liveness": liveness,
    "loudness": loudness,
    "speechiness": speechiness,
    "tempo": tempo,
    "audio_valence": audio_valence,
    "audio_mode": audio_mode,
    "key": key,
    "time_signature": time_signature,
}
input_df = pd.DataFrame([raw_input])

# ---------------------------------------------------------
# 4) Preprocessing di sisi app:
#    - log1p fitur skewed (jika >= 0)
#    - get_dummies untuk kategori
#    - reindex ke expected_cols (isi 0 untuk yang hilang)
#    - (opsional) scaling -> hanya dilakukan bila fitur scaler == fitur model
# ---------------------------------------------------------
for f in numerical_features_for_log:
    if f in input_df.columns:
        input_df[f] = input_df[f].apply(lambda x: np.log1p(x) if x >= 0 else x)

# pastikan kolom kategorikal bertipe string agar nama dummy stabil
for c in categorical_features:
    if c in input_df.columns:
        input_df[c] = input_df[c].astype("string")

X_enc = pd.get_dummies(input_df, columns=categorical_features, drop_first=False)

# reindex ke kolom yang diharapkan model
X_aligned = X_enc.reindex(columns=expected_cols, fill_value=0)

# (opsional) scaling: hanya jalan jika daftar kolom scaler == daftar kolom model
use_scaler = False
if scaler is not None and hasattr(scaler, "feature_names_in_"):
    scaler_cols = list(scaler.feature_names_in_)
    if scaler_cols == expected_cols:
        use_scaler = True

if use_scaler:
    X_arr = scaler.transform(X_aligned)
    X_ready = pd.DataFrame(X_arr, columns=expected_cols, index=X_aligned.index)
else:
    # GradientBoostingRegressor tidak wajib scaling; aman dipakai apa adanya
    X_ready = X_aligned

# ---------------------------------------------------------
# 5) Prediksi + bantuan debug kalau masih error
# ---------------------------------------------------------
if st.button("Prediksi Popularitas"):
    try:
        pred = model.predict(X_ready)
        st.subheader(f"Prediksi Popularitas Lagu: {pred[0]:.2f}")
    except Exception as e:
        # tampilkan mismatch utk debugging cepat
        need = set(expected_cols)
        got  = set(X_ready.columns)
        missing = sorted(list(need - got))
        extra   = sorted(list(got - need))
        if missing:
            st.warning(f"Missing columns (contoh 15 pertama): {missing[:15]}")
        if extra:
            st.info(f"Extra columns (contoh 15 pertama): {extra[:15]}")
        st.error(f"Gagal melakukan prediksi: {e}")

st.markdown("---")
#st.caption(
#    "Jika tetap error, sangat mungkin file model tidak menyertakan feature list. "
 #   "Simpan ulang dari training dengan `bundle['feature_columns'] = X_train.columns.tolist()`."
#)

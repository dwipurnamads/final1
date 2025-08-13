import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

# Function to load the combined pipeline (model and scaler)
@st.cache_resource
def load_pipeline():
    # Load the pipeline from the single file
    with open('full_pipeline.pkl', 'rb') as pipeline_file:
        pipeline = pickle.load(pipeline_file)

    model = pipeline['model']
    scaler = pipeline['scaler']

    return model, scaler

model, scaler = load_pipeline()

# Define the list of numerical features for scaling
numerical_features_for_scaling = ['song_duration_ms', 'acousticness', 'danceability', 'energy',
                                'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo',
                                'audio_valence']

# Define the list of categorical features for one-hot encoding
categorical_features = ['audio_mode', 'key', 'time_signature']

# Streamlit App Title
st.title("Prediksi Popularitas Lagu")

st.markdown("""
Aplikasi ini memprediksi popularitas lagu berdasarkan fitur-fitur audio.
Mohon masukkan nilai untuk setiap fitur di bawah ini.
""")

# Input features from the user
st.header("Masukkan Fitur Lagu")

# Using columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    song_duration_ms = st.number_input("Song Duration (ms)", min_value=0.0, value=200000.0)
    acousticness = st.slider("Acousticness", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    danceability = st.slider("Danceability", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    energy = st.slider("Energy", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    instrumentalness = st.slider("Instrumentalness", min_value=0.0, max_value=1.0, value=0.0, step=0.0001)

with col2:
    liveness = st.slider("Liveness", min_value=0.0, max_value=1.0, value=0.1, step=0.001)
    loudness = st.slider("Loudness (dB)", min_value=-60.0, max_value=0.0, value=-10.0, step=0.1)
    speechiness = st.slider("Speechiness", min_value=0.0, max_value=1.0, value=0.05, step=0.001)
    tempo = st.number_input("Tempo (bpm)", min_value=0.0, value=120.0)
    audio_valence = st.slider("Audio Valence", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

with col3:
    audio_mode = st.radio("Audio Mode", options=[0, 1], format_func=lambda x: "Minor" if x == 0 else "Major")
    key = st.selectbox("Key", options=list(range(12)))
    time_signature = st.selectbox("Time Signature", options=[0, 1, 3, 4, 5])

# Create a dictionary from the input values
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

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Apply log transformation to skewed numerical features (as done in preprocessing)
# Need to handle potential zero values before log transform
skewed_features = ['song_duration_ms', 'acousticness', 'instrumentalness', 'liveness', 'speechiness', 'tempo']
for feature in skewed_features:
    if feature in input_df.columns:
        input_df[feature] = np.log1p(input_df[feature]) # log1p handles 0 values

# Apply One-Hot Encoding to categorical features
# Need to ensure all possible columns from training are present, fill with 0 if not
# Note: scaler.feature_names_in_ holds the column names AFTER encoding and scaling from training data
encoded_cols_train = scaler.feature_names_in_
input_df_encoded = pd.get_dummies(input_df, columns=categorical_features, drop_first=True)

# Align columns with the training data - add missing columns and fill with 0
missing_cols = set(encoded_cols_train) - set(input_df_encoded.columns)
for c in missing_cols:
    input_df_encoded[c] = 0
# Ensure the order of columns is the same as during training
input_df_encoded = input_df_encoded[encoded_cols_train]


# Apply Standard Scaling to numerical features
# Select only the numerical columns before scaling
numerical_cols_input = input_df_encoded[numerical_features_for_scaling].columns
input_df_encoded[numerical_cols_input] = scaler.transform(input_df_encoded[numerical_cols_input])


# Make prediction
if st.button("Prediksi Popularitas"):
    prediction = model.predict(input_df_encoded)
    st.subheader(f"Prediksi Popularitas Lagu: {prediction[0]:.2f}")

st.markdown("---")
st.markdown("Catatan: Prediksi ini didasarkan hanya pada fitur audio yang tersedia. Faktor eksternal dapat sangat mempengaruhi popularitas lagu.")

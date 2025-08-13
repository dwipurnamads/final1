import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

# Function to load the combined pipeline (model and scaler)
@st.cache_resource
def load_pipeline():
    # Load the pipeline from the single file
    try:
        with open('full_pipeline.pkl', 'rb') as pipeline_file:
            pipeline = pickle.load(pipeline_file)

        model = pipeline['model']
        scaler = pipeline['scaler']

        # Get the feature names the scaler was fit on. This is the expected order and names
        # for the input data after all preprocessing steps (log transform, one-hot encoding, but BEFORE scaling)
        # Let's store this list as it's crucial for alignment
        expected_feature_names_before_scaling = list(scaler.feature_names_in_)


        return model, scaler, expected_feature_names_before_scaling
    except FileNotFoundError:
        st.error("File model tidak ditemukan. Pastikan 'model_deployment/full_pipeline.pkl' ada di direktori yang benar.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        st.stop()


model, scaler, expected_feature_names_before_scaling = load_pipeline()

# Define the list of numerical features for scaling
# Ensure these match the ones used during training
numerical_features_for_scaling = ['song_duration_ms', 'acousticness', 'danceability', 'energy',
                                'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo',
                                'audio_valence']

# Define the list of categorical features for one-hot encoding
# Ensure these match the ones used during training
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
        # Add a small value before log transformation if the value is 0
        input_df[feature] = input_df[feature].apply(lambda x: np.log1p(x) if x > 0 else 0)


# --- Start of Modified One-Hot Encoding and Column Alignment ---

# Apply One-Hot Encoding to the input data
# Use get_dummies on the categorical columns
# We do NOT use drop_first=True here initially, as we will handle the dropping/alignment
# based on the expected columns from training data
input_df_encoded = pd.get_dummies(input_df, columns=categorical_features, drop_first=False)


# Now, reindex the encoded input DataFrame to match the expected feature names from training data.
# expected_feature_names_before_scaling contains the exact list of columns (including encoded ones)
# that the scaler was fit on, which is the structure expected by the model after encoding but before scaling.
input_df_aligned_before_scaling = input_df_encoded.reindex(columns=expected_feature_names_before_scaling, fill_value=0)

# --- End of Modified One-Hot Encoding and Column Alignment ---


# Apply Standard Scaling to numerical features
# Identify which columns in the aligned DataFrame are the numerical features that need scaling
# We use the original list of numerical features for this
numerical_cols_to_scale = [col for col in numerical_features_for_scaling if col in input_df_aligned_before_scaling.columns]

# Apply scaling ONLY to these numerical columns in the aligned DataFrame
# Create a copy to avoid SettingWithCopyWarning
input_df_aligned_scaled = input_df_aligned_before_scaling.copy()
input_df_aligned_scaled[numerical_cols_to_scale] = scaler.transform(input_df_aligned_before_scaling[numerical_cols_to_scale])


# Make prediction
if st.button("Prediksi Popularitas"):
    # The input_df_aligned_scaled DataFrame should now have the correct columns, order, and scaling
    # It should be ready for the model
    prediction = model.predict(input_df_aligned_scaled)
    st.subheader(f"Prediksi Popularitas Lagu: {prediction[0]:.2f}")

st.markdown("---")
st.markdown("Catatan: Prediksi ini didasarkan hanya pada fitur audio yang tersedia. Faktor eksternal dapat sangat mempengaruhi popularitas lagu.")

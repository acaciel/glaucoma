import streamlit as st
import numpy as np
import cv2
import joblib
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# Load model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ========================
# Feature Extraction
# ========================

def extract_glcm_features(image):
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    glcm = graycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)

    features = []
    for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
        features.extend(graycoprops(glcm, prop).flatten())

    return features

def extract_lbp_features(image, P, R):
    lbp = local_binary_pattern(image, P, R, method="uniform")
    n_bins = int(lbp.max() + 1)

    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return hist

def extract_features(image):
    features = []

    # GLCM
    features.extend(extract_glcm_features(image))

    # LBP Multiscale
    lbp_params = [(8,1), (16,2)]
    for P, R in lbp_params:
        features.extend(extract_lbp_features(image, P, R))

    return np.array(features)

# ========================
# UI
# ========================

st.title("Deteksi Glaukoma dari Citra Fundus")
st.write("Upload citra fundus untuk klasifikasi Normal atau Glaukoma")

uploaded_file = st.file_uploader("Upload gambar...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Citra Input", use_column_width=True)

    # Preprocessing
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (256,256))

    # Feature extraction
    features = extract_features(img_gray)
    features = scaler.transform([features])

    # Prediction
    prediction = model.predict(features)[0]

    if prediction == 0:
        st.success("Hasil: Normal")
    else:
        st.error("Hasil: Glaukoma")
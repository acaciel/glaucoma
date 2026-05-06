import streamlit as st
import numpy as np
import cv2
import joblib
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# ========================
# Load Model & Scaler
# ========================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ========================
# Feature Extraction
# ========================

def extract_lbp_features(image, P, R):
    lbp = local_binary_pattern(image, P, R, method="uniform")
    
    # FIXED bin (WAJIB untuk konsistensi)
    n_bins = P + 2  
    
    hist, _ = np.histogram(
        lbp,
        bins=n_bins,
        range=(0, n_bins),
        density=True
    )
    
    return hist


def extract_glcm_features(image):
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    glcm = graycomatrix(
        image,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True
    )

    features = []
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

    for prop in properties:
        feat = graycoprops(glcm, prop)
        features.extend(feat.flatten())

    return features


def extract_features(image):
    features = []

    # ========================
    # LBP dulu (SAMA kayak training)
    # ========================
    lbp_params = [(8,1), (16,2)]
    for P, R in lbp_params:
        features.extend(extract_lbp_features(image, P, R))

    # ========================
    # Baru GLCM
    # ========================
    features.extend(extract_glcm_features(image))

    return np.array(features)


# ========================
# UI Streamlit
# ========================

st.title("Deteksi Glaukoma dari Citra Fundus")
st.write("Upload citra fundus untuk diklasifikasikan menjadi Normal atau Glaukoma")

uploaded_file = st.file_uploader("Upload gambar...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Citra Input", use_column_width=True)

    # ========================
    # Preprocessing
    # ========================
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (256, 256))

    # ========================
    # Feature Extraction
    # ========================
    features = extract_features(img_gray)

    # DEBUG (boleh hapus nanti)
    st.write("Jumlah fitur:", len(features))

    # ========================
    # Scaling
    # ========================
    features = scaler.transform([features])

    # ========================
    # Prediction
    # ========================
    prediction = model.predict(features)[0]

    if prediction == 0:
        st.success("Hasil: Normal")
    else:
        st.error("Hasil: Glaukoma")

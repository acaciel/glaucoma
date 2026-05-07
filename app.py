import streamlit as st
import numpy as np
import cv2
import joblib
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# ========================
# Page Config
# ========================
st.set_page_config(
    page_title="Deteksi Glaukoma",
    page_icon="👁️",
    layout="centered"
)

# ========================
# Custom CSS
# ========================
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}

.title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    color: #1f3b73;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555555;
    margin-bottom: 30px;
}

.info-box {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

.result-normal {
    background-color: #d4edda;
    color: #155724;
    padding: 15px;
    border-radius: 12px;
    font-size: 20px;
    font-weight: bold;
    text-align: center;
}

.result-glaucoma {
    background-color: #f8d7da;
    color: #721c24;
    padding: 15px;
    border-radius: 12px;
    font-size: 20px;
    font-weight: bold;
    text-align: center;
}
""", unsafe_allow_html=True)

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

    # ===
    # LBP
    # ===
    lbp_params = [(8,1), (16,2)]
    for P, R in lbp_params:
        features.extend(extract_lbp_features(image, P, R))

    # ====
    # GLCM
    # ====
    features.extend(extract_glcm_features(image))

    return np.array(features)

# ========================
# Header
# ========================

st.markdown('<p class="title">👁️ Deteksi Glaukoma</p>', unsafe_allow_html=True)

st.markdown(
    '<p class="subtitle">Klasifikasi citra fundus menggunakan kombinasi fitur GLCM dan Multiscale LBP dengan algoritma SVM</p>',
    unsafe_allow_html=True
)

# ========================
# Information Box
# ========================

st.markdown("""
<div class="info-box">
<b>Informasi Sistem</b><br><br>
Sistem ini digunakan untuk melakukan klasifikasi citra fundus retina ke dalam kategori <b>Normal</b> atau <b>Glaukoma</b> menggunakan metode ekstraksi fitur tekstur dan algoritma Support Vector Machine (SVM).
</div>
""", unsafe_allow_html=True)

# ========================
# Upload Section
# ========================

uploaded_file = st.file_uploader(
    "Upload citra fundus retina",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Citra Fundus Input", use_container_width=True)

    # ========================
    # Preprocessing
    # ========================

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (256, 256))

    # ========================
    # Feature Extraction
    # ========================

    features = extract_features(img_gray)

    # Scaling
    features = scaler.transform([features])

    # Prediction
    prediction = model.predict(features)[0]

    st.markdown("<br>", unsafe_allow_html=True)

    # ========================
    # Result Display
    # ========================

    if prediction == 0:
        st.markdown(
            '<div class="result-normal">✅ Hasil Prediksi: NORMAL</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-glaucoma">⚠️ Hasil Prediksi: GLAUKOMA</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.info("Model menggunakan kombinasi fitur GLCM dan Multiscale LBP dengan klasifikasi SVM")

# ========================
# Footer
# ========================

st.markdown("---")
st.caption("Skripsi Deteksi Glaukoma Menggunakan Kombinasi GLCM dan Multiscale LBP")

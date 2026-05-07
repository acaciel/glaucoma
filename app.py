import streamlit as st
from PIL import Image
import numpy as np
import cv2
import joblib
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# ========================
# PAGE CONFIG
# ========================

st.set_page_config(
    page_title="Deteksi Glaukoma",
    page_icon="👁️",
    layout="centered"
)

# ========================
# CUSTOM CSS
# ========================

st.markdown("""
<style>

.stApp {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 800px;
}

h1 {
    text-align: center;
    color: #1f2937;
}

.subtitle {
    text-align: center;
    color: #6b7280;
    margin-bottom: 2rem;
    font-size: 17px;
}

.result-box {
    padding: 1rem;
    border-radius: 16px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    margin-top: 1rem;
}

.normal {
    background-color: #dcfce7;
    color: #166534;
}

.glaucoma {
    background-color: #fee2e2;
    color: #991b1b;
}

</style>
""", unsafe_allow_html=True)

# ========================
# LOAD MODEL
# ========================

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ========================
# FEATURE EXTRACTION
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

    properties = [
        'contrast',
        'dissimilarity',
        'homogeneity',
        'energy',
        'correlation'
    ]

    for prop in properties:
        feat = graycoprops(glcm, prop)
        features.extend(feat.flatten())

    return features

def extract_features(image):
    features = []

    lbp_params = [(8,1), (16,2)]

    # LBP dulu
    for P, R in lbp_params:
        features.extend(extract_lbp_features(image, P, R))

    # Baru GLCM
    features.extend(extract_glcm_features(image))

    return np.array(features)

# ========================
# HEADER
# ========================

st.title("👁️ Deteksi Glaukoma")

st.markdown(
    '<div class="subtitle">Klasifikasi citra fundus retina menggunakan kombinasi GLCM, MS-LBP, dan SVM</div>',
    unsafe_allow_html=True
)

# ========================
# UPLOAD
# ========================

uploaded_file = st.file_uploader(
    "Upload citra fundus...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Citra Fundus",
        use_container_width=True
    )

    if st.button("🔍 Analisis Gambar"):

        with st.spinner("Sedang memproses gambar..."):

            # Convert image
            img = np.array(image)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (256, 256))

            # Feature extraction
            features = extract_features(img)

            # Scaling
            features = scaler.transform([features])

            # Prediction
            prediction = model.predict(features)[0]

            # Result
            if prediction == 0:
                st.markdown(
                    '<div class="result-box normal">✅ NORMAL</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="result-box glaucoma">⚠️ GLAUKOMA</div>',
                    unsafe_allow_html=True
                )

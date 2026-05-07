# Upgrade UI Streamlit App — Modern Version

Ganti seluruh isi `app.py` dengan kode berikut untuk mendapatkan tampilan yang lebih modern, clean, dan tidak terlalu kaku.

```python
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
    page_icon="🩺",
    layout="wide"
)

# ========================
# Custom CSS
# ========================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to bottom right, #eef4ff, #f8fbff);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.main-title {
    font-size: 48px;
    font-weight: 800;
    color: #1b2559;
    margin-bottom: 10px;
}

.subtitle {
    font-size: 18px;
    color: #5b6478;
    line-height: 1.7;
}

.glass-card {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(12px);
    padding: 28px;
    border-radius: 24px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    border: 1px solid rgba(255,255,255,0.3);
}

.metric-card {
    background: white;
    padding: 18px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}

.metric-title {
    color: #6c7293;
    font-size: 15px;
}

.metric-value {
    color: #1b2559;
    font-size: 24px;
    font-weight: 700;
}

.result-normal {
    background: linear-gradient(135deg, #d4fc79, #96e6a1);
    padding: 22px;
    border-radius: 22px;
    color: #134b2f;
    font-size: 26px;
    font-weight: 700;
    text-align: center;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}

.result-glaucoma {
    background: linear-gradient(135deg, #ff9a9e, #fecfef);
    padding: 22px;
    border-radius: 22px;
    color: #6e1c29;
    font-size: 26px;
    font-weight: 700;
    text-align: center;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}

.footer {
    text-align: center;
    color: #7d8597;
    margin-top: 30px;
    font-size: 14px;
}

</style>
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

    lbp_params = [(8,1), (16,2)]

    # LBP dulu
    for P, R in lbp_params:
        features.extend(extract_lbp_features(image, P, R))

    # Baru GLCM
    features.extend(extract_glcm_features(image))

    return np.array(features)

# ========================
# Layout
# ========================

left_col, right_col = st.columns([1.1, 1])

with left_col:

    st.markdown('<div class="main-title">👁️ Deteksi Glaukoma</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="subtitle">Sistem klasifikasi citra fundus retina menggunakan kombinasi fitur <b>Gray Level Co-occurrence Matrix (GLCM)</b> dan <b>Multiscale Local Binary Pattern (MS-LBP)</b> dengan algoritma <b>Support Vector Machine (SVM)</b>.</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
    <h4>📌 Tentang Sistem</h4>
    Sistem ini dirancang untuk membantu proses identifikasi awal glaukoma melalui analisis tekstur citra fundus retina secara otomatis.
    <br><br>
    Model klasifikasi dibangun menggunakan kombinasi fitur tekstur dan algoritma machine learning untuk membedakan citra retina normal dan glaukoma.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('''
        <div class="metric-card">
            <div class="metric-title">Akurasi</div>
            <div class="metric-value">95.7%</div>
        </div>
        ''', unsafe_allow_html=True)

    with col2:
        st.markdown('''
        <div class="metric-card">
            <div class="metric-title">Metode</div>
            <div class="metric-value">SVM</div>
        </div>
        ''', unsafe_allow_html=True)

    with col3:
        st.markdown('''
        <div class="metric-card">
            <div class="metric-title">Fitur</div>
            <div class="metric-value">GLCM + MS-LBP</div>
        </div>
        ''', unsafe_allow_html=True)

with right_col:

    st.markdown("<br>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📤 Upload Citra Fundus",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        st.image(img, caption="Citra Fundus Retina", use_container_width=True)

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

        if prediction == 0:
            st.markdown(
                '<div class="result-normal">✅ NORMAL</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-glaucoma">⚠️ GLAUKOMA</div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        st.success("Prediksi berhasil dilakukan menggunakan model klasifikasi SVM")

# ========================
# Footer
# ========================

st.markdown(
    '<div class="footer">Skripsi — Deteksi Glaukoma Menggunakan Kombinasi Fitur GLCM dan Multiscale LBP</div>',
    unsafe_allow_html=True
)
```

import streamlit as st
    st.markdown("""
    <div class="glass-card">
    <h4>📌 Tentang Sistem</h4>
    Sistem ini dirancang untuk membantu proses identifikasi awal glaukoma melalui analisis tekstur citra fundus retina secara otomatis.
    <br><br>
    Model klasifikasi dibangun menggunakan kombinasi fitur tekstur dan algoritma machine learning untuk membedakan citra retina normal dan glaukoma.
    </div>
    """, unsafe_allow_html=True)

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

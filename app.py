import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans

# --------------------- PAGE CONFIG & STYLING ---------------------
st.set_page_config(page_title="🎓 Dropout Risk Prediction", layout="wide")

st.markdown("""
    <style>
    html, body, .stApp {
        background-color: #f8f9fa !important;
        color: #212529 !important;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3, h4 {
        color: #0d6efd !important;
    }
    .stButton>button {
        background-color: #0d6efd !important;
        color: white !important;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
    }
    .stDataFrame, .stTable {
        background-color: white !important;
        color: #212529 !important;
    }
    .block-container {
        padding: 2rem 3rem;
    }
    .css-1d391kg, .css-1v0mbdj {
        padding: 2rem 2rem;
    }
    .stSidebar {
        background-color: #ffffff !important;
        border-right: 1px solid #dee2e6;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------- LOAD DATA ---------------------
@st.cache_data
def load_data():
    return pd.read_csv("students_dropout_academic_success_terbaru.csv")

df = load_data()

# --------------------- MAPPING KATEGORI ---------------------
category_mappings = {
    'Marital Status': {
        1: 'Single',
        2: 'Married',
        3: 'Widower',
        4: 'Divorced',
        5: 'Facto union',
        6: 'Legally separated'
    },
    'Daytime/evening attendance': {
        1: 'Daytime',
        2: 'Evening'
    },
    'Gender': {
        1: 'Female',
        2: 'Male'
    },
    'Scholarship holder': {
        1: 'No',
        2: 'Yes'
    },
    'Displaced': {
        1: 'No',
        2: 'Yes'
    },
    'Debtor': {
        1: 'No',
        2: 'Yes'
    },
    'Educational special needs': {
        1: 'No',
        2: 'Yes'
    },
    'International': {
        1: 'No',
        2: 'Yes'
    },
    'Tuition fees up to date': {
        1: 'Yes',
        2: 'No'
    }
}

try:
    model = joblib.load("model_dropout.pkl")
    scaler = joblib.load("scaler_dropout.pkl")
except:
    model, scaler = None, None

# --------------------- SIDEBAR ---------------------
st.sidebar.image("https://img.freepik.com/free-vector/education-logo-template_1061-28.jpg", width=200)
menu = st.sidebar.radio("Navigasi", [
    "🏠 Home", "📈 Business Understanding", "🔍 Data Understanding",
    "📊 Data Exploration", "🧹 Data Preparation",
    "📉 Modeling & Evaluation", "🧪 Prediksi Dropout", "📍 Clustering Mahasiswa"
])

# --------------------- HOME ---------------------
if menu == "🏠 Home":
    st.markdown("""
    <div style='text-align:center;'>
        <h1>DASHBOARD DETEKSI RISIKO DROPOUT MAHASISWA</h1>
        <p style='font-size:18px;'>"Analisis Risiko Dropout Mahasiswa Berdasarkan Data Akademik dan Sosial"</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("👥 Anggota Kelompok 7 - SI4709"):
        anggota = {
            "Aurellia Verly": "102022330371",
            "Bagas Haris Saputro": "102022300291",
            "Cheisya Valda Wibawaningrum": "102022330316",
            "Naufal Athalino Bakti": "102022300239"
        }
        st.table(pd.DataFrame(anggota.items(), columns=["Nama", "NIM"]))

# --------------------- BUSINESS UNDERSTANDING ---------------------
elif menu == "📈 Business Understanding":
    st.header("Business Understanding")

    st.markdown("""
    ### 🔵 Risiko Dropout
    Risiko dropout merupakan ancaman signifikan dalam dunia pendidikan tinggi yang dapat memengaruhi efektivitas pembelajaran, alokasi sumber daya, dan citra institusi.
    Mahasiswa yang keluar tanpa menyelesaikan studinya dapat disebabkan oleh banyak faktor, termasuk tekanan akademik, sosial, hingga kondisi ekonomi.

    > Risiko dropout dapat menghambat keberhasilan akademik dan efisiensi operasional universitas. Oleh karena itu, sangat penting untuk melakukan deteksi dini dan intervensi.
    """)

    st.markdown("""
    ### 🎯 Tujuan Proyek
    Proyek ini bertujuan untuk membangun sistem prediksi berbasis machine learning yang mampu mengidentifikasi mahasiswa dengan risiko tinggi untuk dropout.
    Model klasifikasi digunakan untuk mengolah data akademik dan demografis mahasiswa seperti nilai, status pembayaran, usia, jenis kelamin, dan lainnya.

    > Dengan sistem ini, diharapkan institusi dapat mengurangi angka dropout melalui intervensi yang lebih terarah dan akurat.
    """)

    st.markdown("""
    ### 🔍 Aspek Penting Proyek
    - **Identifikasi Risiko**: Menganalisis pola historis yang berkaitan dengan perilaku dropout mahasiswa.
    - **Optimalisasi Intervensi**: Mengarahkan sumber daya ke mahasiswa yang benar-benar membutuhkan.
    - **Pengambilan Keputusan**: Memberikan insight berbasis data kepada manajemen kampus.

    > Pemahaman mendalam terhadap risiko dropout membantu institusi mengelola populasi mahasiswa secara lebih efektif.
    """)

    st.markdown("""
    ### 🧩 Manfaat untuk Institusi dan Stakeholder
    - **Kampus**: Meningkatkan retensi, efisiensi beasiswa, dan perencanaan akademik.
    - **Mahasiswa**: Mendapatkan bantuan dan perhatian lebih awal sebelum risiko meningkat.
    - **Dosen & Penasihat Akademik**: Memiliki alat bantu berbasis data untuk memberikan bimbingan yang lebih akurat.
    """)

    st.markdown("""
    ### 📊 Nilai Keberhasilan Model
    Keberhasilan sistem ini dievaluasi berdasarkan:
    - **Akurasi Model**: Seberapa tepat model mengklasifikasikan mahasiswa yang dropout dan tidak.
    - **Recall & F1-Score**: Seberapa baik sistem mengenali mahasiswa yang benar-benar dropout.
    - **Kemampuan Generalisasi**: Adaptabilitas model terhadap data dari berbagai jurusan dan angkatan.
    """)

# --------------------- DATA UNDERSTANDING ---------------------
elif menu == "🔍 Data Understanding":
    st.title("Data Understanding")
    st.subheader("🔹 Informasi Umum Dataset")
    st.write(f"Dataset terdiri dari **{df.shape[0]} baris** dan **{df.shape[1]} kolom**.")
    st.subheader("🔹 Tipe Data")
    st.dataframe(df.dtypes.reset_index().rename(columns={0: "Tipe", "index": "Kolom"}))
    st.subheader("🔹 Missing Values")
    missing = df.isnull().sum()
    st.dataframe(missing[missing > 0].reset_index().rename(columns={0: "Jumlah Missing", "index": "Kolom"}))

# --------------------- DATA EXPLORATION ---------------------
elif menu == "📊 Data Exploration":
    st.title("Eksplorasi Data")
    st.subheader("🔸 Korelasi dengan Target")
    st.dataframe(df.corr(numeric_only=True)['target'].sort_values(ascending=False))
    st.subheader("🔸 Heatmap Korelasi")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm', annot=False, ax=ax)
    st.pyplot(fig)

# --------------------- DATA PREPARATION ---------------------
elif menu == "🧹 Data Preparation":
    st.title("Data Preparation")
    st.markdown("""
    Data dibagi menjadi training dan testing, lalu dilakukan normalisasi dan penyeimbangan (SMOTE).
    """)
    X = df.drop(columns='target')
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler_prep = StandardScaler()
    X_train_scaled = scaler_prep.fit_transform(X_train)
    smote = SMOTE()
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    st.success("Data siap untuk pelatihan model.")

    st.code("""
# Split data
dari sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scaling
dari sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE()
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
    """, language="python")

# --------------------- MODELING & EVALUATION ---------------------
elif menu == "📉 Modeling & Evaluation":
    st.header("Modeling & Evaluation")
    if model:
        X_scaled = scaler.transform(df.drop(columns='target'))
        y = df['target']
        y_pred = model.predict(X_scaled)

        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        # Simpan ke session_state
        st.session_state["acc"] = acc
        st.session_state["prec"] = prec
        st.session_state["rec"] = rec
        st.session_state["f1"] = f1

        # Tampilkan metrik
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Akurasi", f"{acc:.2f}")
        col2.metric("Precision", f"{prec:.2f}")
        col3.metric("Recall", f"{rec:.2f}")
        col4.metric("F1 Score", f"{f1:.2f}")

        option = st.selectbox("Pilih Visualisasi Evaluasi", ["Confusion Matrix", "ROC Curve"])
        if option == "Confusion Matrix":
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt='d', cmap='Greens', ax=ax)
            st.pyplot(fig)
        else:
            y_prob = model.predict_proba(X_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_prob)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}", color='blue')
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            st.pyplot(fig)
    else:
        st.warning("Model belum tersedia.")

# --------------------- PREDIKSI ---------------------
elif menu == "🧪 Prediksi Dropout":
    st.header("Prediksi Risiko Dropout Mahasiswa")

    if model and scaler:
        with st.form("form_prediksi"):
            input_data = {}

            def dropdown_kategori(col, keterangan, options):
                st.markdown(f"**{col}**")
                for k, v in keterangan.items():
                    st.markdown(f"{k} = {v}")
                selected = st.selectbox(f"Pilih nilai untuk {col}", options, key=col)
                return int(selected)

            kategori_fields = {
                "Marital Status": {1: "Single", 2: "Married", 3: "Widower", 4: "Divorced", 5: "Facto Union", 6: "Legally Separated"},
                "Application order": {0: "1st Choice", 1: "2nd Choice", 2: "3rd", 3: "4th", 4: "5th", 5: "6th", 6: "7th"},
                "Daytime/evening attendance": {0: "Evening", 1: "Daytime"},
                "Displaced": {0: "No", 1: "Yes"},
                "Educational special needs": {0: "No", 1: "Yes"},
                "Debtor": {0: "No", 1: "Yes"},
                "Tuition fees up to date": {0: "No", 1: "Yes"},
                "Gender": {0: "Female", 1: "Male"},
                "Scholarship holder": {0: "No", 1: "Yes"},
                "International": {0: "No", 1: "Yes"},
                "Curricular units 2nd sem (without evaluations)": {i: f"{i}" for i in range(13)},
                "Unemployment rate": {i+1: val for i, val in enumerate(sorted(df["Unemployment rate"].dropna().unique()))},
                "Inflation rate": {i+1: val for i, val in enumerate(sorted(df["Inflation rate"].dropna().unique()))},
                "GDP": {i+1: val for i, val in enumerate(sorted(df["GDP"].dropna().unique()))},
            }

            for col in df.drop(columns='target').columns:
                if col in kategori_fields:
                    keterangan = kategori_fields[col]
                    input_data[col] = dropdown_kategori(col, keterangan, list(keterangan.keys()))

                elif col in [
                    'Previous qualification (grade)',
                    'Admission grade',
                    'Curricular units 1st sem (grade)',
                    'Curricular units 2nd sem (grade)'
                ]:
                    input_data[col] = st.number_input(f"{col} (range: 0.0 - 190.0)", min_value=0.0, max_value=190.0, step=0.1)

                elif col == "Application mode":
                    input_data[col] = st.number_input(f"{col} (range: 1 - 57)", min_value=1, max_value=57, step=1)

                elif col == "Course":
                    input_data[col] = st.number_input(f"{col} (range: 33 - 9991)", min_value=33, max_value=9991, step=1)

                elif col == "Previous qualification":
                    input_data[col] = st.number_input(f"{col} (range: 1 - 43)", min_value=1, max_value=43, step=1)

                elif col == "Nacionality":
                    input_data[col] = st.number_input(f"{col} (range: 1 - 109)", min_value=1, max_value=109, step=1)

                elif col in ["Mother's qualification", "Father's qualification"]:
                    input_data[col] = st.number_input(f"{col} (range: 1 - 44)", min_value=1, max_value=44, step=1)

                elif col == "Mother's occupation":
                    input_data[col] = st.number_input(f"{col} (range: 0 - 194)", min_value=0, max_value=194, step=1)

                elif col == "Father's occupation":
                    input_data[col] = st.number_input(f"{col} (range: 0 - 195)", min_value=0, max_value=195, step=1)

                elif col == "Age at enrollment":
                    input_data[col] = st.number_input(f"{col} (range: 17 - 70)", min_value=17, max_value=70, step=1)

                elif "1st sem" in col or "2nd sem" in col:
                    if "grade" not in col:
                        min_max = {
                            'Curricular units 1st sem (credited)': (0, 20),
                            'Curricular units 1st sem (enrolled)': (0, 26),
                            'Curricular units 1st sem (evaluations)': (0, 45),
                            'Curricular units 1st sem (approved)': (0, 26),
                            'Curricular units 1st sem (without evaluations)': (0, 12),
                            'Curricular units 2nd sem (credited)': (0, 19),
                            'Curricular units 2nd sem (enrolled)': (0, 23),
                            'Curricular units 2nd sem (evaluations)': (0, 33),
                            'Curricular units 2nd sem (approved)': (0, 20),
                        }
                        min_val, max_val = min_max.get(col, (0, 100))
                        input_data[col] = st.number_input(f"{col} (range: {min_val} - {max_val})", min_value=min_val, max_value=max_val, step=1)
                    else:
                        input_data[col] = st.number_input(f"{col} (range: 0.0 - 20.0)", min_value=0.0, max_value=20.0, step=0.1)

                else:
                    st.write(f"{col} belum ditangani khusus.")  # debug atau reminder

            submit = st.form_submit_button("🔍 Prediksi")

            if submit:
                input_df = pd.DataFrame([input_data])
                input_scaled = scaler.transform(input_df)
                pred = model.predict(input_scaled)[0]
                st.success(f"Hasil Prediksi: {'BERPOTENSI DROPOUT' if pred==1 else 'TIDAK DROPOUT'}")

                if "acc" in st.session_state:
                    st.markdown("---")
                    st.subheader("📊 Model Evaluation Metrics (Dataset)")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Akurasi", f"{st.session_state['acc']:.2f}")
                    col2.metric("Precision", f"{st.session_state['prec']:.2f}")
                    col3.metric("Recall", f"{st.session_state['rec']:.2f}")
                    col4.metric("F1 Score", f"{st.session_state['f1']:.2f}")
                else:
                    st.warning("Silakan jalankan Modeling & Evaluation terlebih dahulu untuk melihat metrik.")

    else:
        st.warning("Model dan scaler belum dimuat.")

# --------------------- CLUSTERING ---------------------
elif menu == "📍 Clustering Mahasiswa":
    st.header("Segmentasi Mahasiswa Berdasarkan Clustering")
    num_cluster = st.slider("Jumlah Cluster (K)", min_value=2, max_value=10, value=3)
    X_cluster = df.drop(columns='target')
    X_scaled = scaler.transform(X_cluster)

    kmeans = KMeans(n_clusters=num_cluster, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    st.subheader("Distribusi Mahasiswa per Cluster")
    st.bar_chart(df['cluster'].value_counts().sort_index())

    st.subheader("Visualisasi 2D (PCA)")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df['pca1'] = pca_result[:, 0]
    df['pca2'] = pca_result[:, 1]
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='Set2', ax=ax)
    st.pyplot(fig)

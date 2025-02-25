import pandas as pd
import streamlit as st
import joblib
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split  # Tambahkan import untuk train_test_split

# Lakukan unduhan NLTK di awal skrip
nltk.download('stopwords')
nltk.download('punkt')  # Unduh tokenizer 'punkt' untuk bahasa Indonesia

# Membaca model yang sudah dilatih
logreg_model = joblib.load("model100.pkl")

# Inisialisasi objek TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Membaca data dari file CSV
df = pd.read_csv("data.csv")

# Memisahkan fitur (X) dan label (y)
X = df['Text']
y = df['Human']

# Memisahkan data menjadi data pelatihan (training) dan data pengujian (testing) dengan rasio 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi objek TF-IDF Vectorizer dan melakukan fit_transform pada data pelatihan
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Fungsi untuk membersihkan teks
def clean_text(text):
    stop_words = set(stopwords.words('indonesian'))
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = text.lower()  # Case folding
    words = word_tokenize(text)  # Tokenizing
    cleaned_words = [word for word in words if word not in stop_words]  # Stopword removal
    stemmed_words = [stemmer.stem(word) for word in cleaned_words]  # Stemming
    return " ".join(stemmed_words)

# Fungsi untuk melakukan klasifikasi teks
def classify_text(input_text):
    # Membersihkan teks input
    cleaned_text = clean_text(input_text)
    # Mengubah teks input menjadi vektor fitur menggunakan TF-IDF
    input_vector = tfidf_vectorizer.transform([cleaned_text])
    # Melakukan prediksi menggunakan model
    predicted_label = logreg_model.predict(input_vector)[0]
    return predicted_label

# Streamlit UI
st.title("Aplikasi Analisis Sentimen Moris")
input_text = st.text_input("Masukkan kalimat untuk analisis sentimen:")
if st.button("Analisis"):
    result = classify_text(input_text)
    st.write("Hasil Analisis Sentimen:", result)

# Tombol menuju website
back_button = st.markdown("""
    <style>
        .button {
            display: inline-block;
            padding: 0.25em 0.75em;
            font-size: 1em;
            font-weight: 400;
            color: #fff;
            background-color: #007bff;
            border: 1px solid transparent;
            border-radius: 0.25rem;
            cursor: pointer;
            text-align: center;
            vertical-align: middle;
            user-select: none;
        }
    </style>
    <a href="https://analisissentimenparfummorris.netlify.app/" target="_blank">
        <button class="button">Back</button>
    </a>
""", unsafe_allow_html=True)
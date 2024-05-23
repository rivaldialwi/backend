import pandas as pd
import re
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import joblib
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from wordcloud import WordCloud

#2
# Membaca dataset dari file CSV
df = pd.read_csv('data.csv')

# Menampilkan nama-nama kolom dalam DataFrame
print(df.columns)

#3
# Membaca dataset
data = pd.read_csv('data.csv')  # Ganti 'nama_file.csv' dengan nama file dataset Anda

# Menampilkan lima baris pertama dari dataset
print("Lima baris pertama dari dataset:")
print(data.head())

# Menampilkan jumlah entri dengan sentimen negatif, netral, dan positif
sentimen_count = data['Human'].value_counts()

# Membuat plot diagram batang
plt.figure(figsize=(8, 6))
sentimen_count.plot(kind='bar', color=['green', 'red', 'grey'])
plt.title('Jumlah Sentimen positif, negatif, dan netral')
plt.xlabel('Sentimen')
plt.ylabel('Jumlah')
plt.xticks(rotation=0)
plt.show()

#4
# Menghitung jumlah sentimen positif, netral, dan negatif
positif_count = (data['Human'] == 'Positif').sum()
netral_count = (data['Human'] == 'Netral').sum()
negatif_count = (data['Human'] == 'Negatif').sum()

# Menampilkan jumlah sentimen positif, netral, dan negatif
print("Jumlah Sentimen Positif:", positif_count)
print("Jumlah Sentimen Netral:", netral_count)
print("Jumlah Sentimen Negatif:", negatif_count)

#5
# PROSES CLEANSING DATA

# Membaca data dari file CSV
df = pd.read_csv("data.csv")

# Membersihkan data kolom 'Human' dari karakter yang tidak diinginkan
def clean_text(text):
    # Contoh: Menghilangkan tanda baca dan mengubah teks menjadi huruf kecil
    text = text.replace(",", "")
    text = text.replace(".", "")
    text = text.replace("!", "")
    text = text.replace("?", "")
    # Lanjutkan sesuai kebutuhan membersihkan teks
    
    return text

# Memanggil fungsi clean_text untuk membersihkan kolom 'Human'
df['Text'] = df['Text'].apply(clean_text)

# Menampilkan hasil setelah membersihkan data
print(df['Text'])

#6
# PROSES CASE FOLDING

# Membaca data dari file CSV
df = pd.read_csv("data.csv")

# Proses case folding pada kolom 'Human'
df['Text'] = df['Text'].str.lower()

# Menampilkan hasil setelah proses case folding
print(df['Text'])

#7
# PROSES Stopword

# Membaca data dari file CSV
df = pd.read_csv("data.csv")

# Mengambil daftar stopword dalam bahasa Indonesia
stopwords_indo = set(stopwords.words('indonesian'))

# Fungsi untuk menghapus stopword dari teks
def remove_stopwords(text):
    words = text.split()  # Memisahkan teks menjadi kata-kata
    filtered_words = [word for word in words if word not in stopwords_indo]
    return ' '.join(filtered_words)

# Memanggil fungsi remove_stopwords untuk menghapus stopword dari kolom 'Human'
df['Text'] = df['Text'].apply(remove_stopwords)

# Menampilkan hasil setelah proses penghapusan stopword
print(df['Text'])

#8
# PROSES Stemming

# Membaca data dari file CSV
df = pd.read_csv("data.csv")

# Inisialisasi stemmer bahasa Inggris
stemmer = PorterStemmer()

# Fungsi untuk melakukan stemming pada teks
def stemming(text):
    words = text.split()  # Memisahkan teks menjadi kata-kata
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

# Memanggil fungsi stemming untuk melakukan stemming pada kolom 'Human'
df['Text'] = df['Text'].apply(stemming)

# Menampilkan hasil setelah proses stemming
print(df['Text'])

#9
# PROSES Tokenizing

# Membaca data dari file CSV
df = pd.read_csv("data.csv")

# Fungsi untuk melakukan tokenisasi pada teks
def tokenize(text):
    tokens = word_tokenize(text)  # Melakukan tokenisasi kata
    return tokens

# Memanggil fungsi tokenize untuk melakukan tokenisasi pada kolom 'Human'
df['Text'] = df['Text'].apply(tokenize)

# Menampilkan hasil setelah proses tokenisasi
print(df['Text'])

#10
# Membaca dataset dari file CSV
df = pd.read_csv('data.csv')

# Membuat word cloud untuk setiap sentimen
for sentiment in df['Human'].unique():
    # Menggabungkan semua teks dalam kolom 'Text' berdasarkan sentimen
    text = ' '.join(df[df['Human'] == sentiment]['Text'])
    
    # Membuat objek WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    # Menampilkan word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud untuk Sentimen {sentiment}')
    plt.axis('off')
    plt.show()

#11
# PROSES Pembobotan Dan Pembagian Data Training Dan Data Testing Menggunakan TF-IDF

# Membaca data dari file CSV
df = pd.read_csv("data.csv")

# Memisahkan fitur (X) dan label (y)
X = df['Text']
y = df['Human']

# Memisahkan data menjadi data pelatihan (training) dan data pengujian (testing) dengan rasio 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi objek TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Melakukan pembelajaran (fitting) dan transformasi pada data pelatihan
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Melakukan transformasi pada data pengujian
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Menampilkan dimensi dari matriks TF-IDF
print("Dimensi matriks TF-IDF untuk data pelatihan:", X_train_tfidf.shape)
print("Dimensi matriks TF-IDF untuk data pengujian:", X_test_tfidf.shape)

#12
# Inisialisasi model regresi logistik multinomial
logreg_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

# Melatih model regresi logistik menggunakan data pelatihan dan labelnya
logreg_model.fit(X_train_tfidf, y_train)

# Memprediksi label untuk data pengujian
y_pred = logreg_model.predict(X_test_tfidf)

#13
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Memprediksi label untuk data pengujian
y_pred = logreg_model.predict(X_test_tfidf)

# Menghitung akurasi prediksi
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi model regresi logistik multinomial:", accuracy)

# Menghitung presisi prediksi
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
print("Presisi model regresi logistik multinomial:", precision)

# Menghitung recall prediksi
recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
print("Recall model regresi logistik multinomial:", recall)

#14
# Melakukan prediksi pada data pelatihan
y_train_pred = logreg_model.predict(X_train_tfidf)

# Melakukan prediksi pada data pengujian
y_test_pred = logreg_model.predict(X_test_tfidf)

# Menghitung metrik evaluasi untuk data pelatihan
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=1)
train_recall = recall_score(y_train, y_train_pred, average='weighted', zero_division=1)

# Menghitung metrik evaluasi untuk data pengujian
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=1)
test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=1)

# Menampilkan hasil evaluasi
print("Hasil evaluasi model pada data pelatihan:")
print(f"Akurasi: {train_accuracy}")
print(f"Presisi: {train_precision}")
print(f"Recall: {train_recall}")

print("\nHasil evaluasi model pada data pengujian:")
print(f"Akurasi: {test_accuracy}")
print(f"Presisi: {test_precision}")
print(f"Recall: {test_recall}")

# Memeriksa kemungkinan overfitting
if train_accuracy > test_accuracy:
    print("\nModel cenderung mengalami overfitting.")
else:
    print("\nModel tidak cenderung mengalami overfitting.")

#15
# Simpan model ke dalam file
joblib.dump(logreg_model, "model100.pkl")

# Output pesan konfirmasi
print("Model berhasil disimpan dalam file 'model100.pkl'.")

#16
# Load model yang sudah dilatih
logreg_model = joblib.load("model100.pkl")

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
    if predicted_label == 'Positif':
        return "Kalimat termasuk dalam kategori: Positif"
    elif predicted_label == 'Negatif':
        return "Kalimat termasuk dalam kategori: Negatif"
    else:
        return "Kalimat termasuk dalam kategori: Netral"

# Contoh penggunaan
input_text = "Perpustakaan ini sangat bagus saya suka belajar di sini tapi buku nya kurang banyak"
result = classify_text(input_text)
print(result)
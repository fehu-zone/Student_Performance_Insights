import pandas as pd
import numpy as np
from scipy import stats 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Veriyi okuyalım
df = pd.read_csv('.\data\StudentPerformanceFactors.csv')

# Sonsuz değerleri NaN olarak değiştirelim (inf ve -inf değerleri)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# İlk birkaç satıra göz atalım
print("İlk 5 Satır:")
print(df.head())

# Verinin genel yapısını kontrol edelim
print("Veri Çerçevesi Bilgileri:")
print(df.info())  # Sütun isimleri, veri tipleri ve null değerler hakkında bilgi

# Kategorik sütunlardaki benzersiz değerleri görmek
categorical_columns = df.select_dtypes(include='object').columns
for col in categorical_columns:
    print(f"{col} sütununun benzersiz değerleri:")
    print(df[col].unique())
    print("-----------")

# Kategorik değişkenleri sayısal değerlere dönüştürme
def encode_column(column, mapping):
    return column.map(mapping)

# Her sütun için benzersiz değerler ve karşılık gelen sayısal değerler
mappings = {
    'Parental_Involvement': {'Low': 0, 'Medium': 1, 'High': 2},
    'Access_to_Resources': {'High': 2, 'Medium': 1, 'Low': 0},
    'Extracurricular_Activities': {'No': 0, 'Yes': 1},
    'Motivation_Level': {'Low': 0, 'Medium': 1, 'High': 2},
    'Internet_Access': {'Yes': 1, 'No': 0},
    'Family_Income': {'Low': 0, 'Medium': 1, 'High': 2},
    'Teacher_Quality': {'Medium': 1, 'High': 2, 'Low': 0},
    'School_Type': {'Public': 0, 'Private': 1},
    'Peer_Influence': {'Positive': 1, 'Negative': 0, 'Neutral': 2},
    'Learning_Disabilities': {'No': 0, 'Yes': 1},
    'Parental_Education_Level': {'High School': 0, 'College': 1, 'Postgraduate': 2},
    'Gender': {'Male': 0, 'Female': 1},
    'Distance_from_Home': {'Near': 0, 'Moderate': 1, 'Far': 2}
}

# Her sütun için kodlama işlemi
for column, mapping in mappings.items():
    df[column] = encode_column(df[column], mapping)

# Sonuçları görüntüleme
print("Kodlanmış Veri Çerçevesinin İlk 5 Satırı:")
print(df.head())

# Veri çerçevesinin temel bilgilerini görüntüleme
print("\nBetimleyici İstatistikler:")
print(df.describe())  # Sayısal sütunların betimleyici istatistikleri

# Null değerlerin kontrolü
print("\nNull Değerlerin Kontrolü:")
print(df.isnull().sum())  # Null değerleri kontrol etme

# Kategorik sütunlar için en sık görülen değeri kullanarak doldurma
df['Teacher_Quality'] = df['Teacher_Quality'].fillna(df['Teacher_Quality'].mode()[0])
df['Parental_Education_Level'] = df['Parental_Education_Level'].fillna(df['Parental_Education_Level'].mode()[0])
df['Distance_from_Home'] = df['Distance_from_Home'].fillna(df['Distance_from_Home'].mode()[0])

# Eksik değerleri kontrol etme
print("\nNull Değerlerin Kontrolü (Doldurma Sonrası):")
print(df.isnull().sum())  # Null değerleri kontrol etme

# Tüm eksik değerlere sahip satırları kaldırma (isteğe bağlı)
# df.dropna(inplace=True)  # Bu satırı kaldırabilirsiniz eğer doldurma işlemi yeterli ise

# Sonuçları yeni bir CSV dosyasına kaydetme
df.to_csv('student_performance_encoded.csv', index=False)
print("Veri, 'student_performance_encoded.csv' olarak kaydedildi.")

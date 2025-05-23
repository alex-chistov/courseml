import pandas as pd
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import s3fs


BUCKET       = "yc-gifts-bucket"
TRAIN_KEY    = "train_data.csv"            
MODEL_KEY    = "models/catboost_model.joblib"
VECT_KEY     = "models/tfidf_vectorizer.joblib"
DATA_PATH    = f"{BUCKET}/{TRAIN_KEY}"

ACCESS_KEY   = ""

# Инициализация S3FileSystem для Yandex Object Storage
fs = s3fs.S3FileSystem(
    key=ACCESS_KEY,
    secret=SECRET_KEY,
    client_kwargs={"endpoint_url": "https://storage.yandexcloud.net"}
)

print("Читаем тренировочный датасет из", DATA_PATH)
with fs.open(DATA_PATH, "rb") as f:
    df = pd.read_csv(f)

print(df.head())
print("Всего строк в train_data.csv:", len(df))

#Векторизация текстов
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["interests"])
y = df["category"]

#Обучение CatBoost
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    verbose=100
)
model.fit(X, y)

joblib.dump(model, "catboost_model.joblib")
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")

print("Загружаем обученную модель и векторизатор в бакет")
with fs.open(f"{BUCKET}/{MODEL_KEY}", "wb") as f:
    joblib.dump(model, f)

with fs.open(f"{BUCKET}/{VECT_KEY}", "wb") as f:
    joblib.dump(vectorizer, f)

print("Обучение завершено.")

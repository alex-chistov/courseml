from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib, pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


VECT_PATH  = "models/tfidf_vectorizer.joblib"
MODEL_PATH = "models/catboost_model.joblib"
GIFTS_PATH = "models/gifts.csv"

vectorizer: TfidfVectorizer = joblib.load(VECT_PATH)
model = joblib.load(MODEL_PATH)

gifts_df: pd.DataFrame = pd.read_csv(GIFTS_PATH)

app = FastAPI(title="ML Gift Recommender")

class Req(BaseModel):
    interests: List[str]

@app.post("/recommend", response_model=List[int])
def recommend(req: Req):
    if not req.interests:
        raise HTTPException(400, "Interests list cannot be empty")

    text = " ".join(req.interests)
    X = vectorizer.transform([text])

    #Предсказываем вероятности для каждой категории
    probs = model.predict_proba(X)[0]
    classes = model.classes_
    top_idx = np.argsort(probs)[::-1][:3]
    top_cats = {classes[i] for i in top_idx}

    # Для каждой из них берём по 3 самых популярных подарка
    ids: List[int] = []
    for cat in top_cats:
        sub = gifts_df[gifts_df["category"] == cat]
        top_gifts = sub.sort_values("popularity", ascending=False).head(3)
        ids.extend(int(i) for i in top_gifts["id"].tolist())

    return ids

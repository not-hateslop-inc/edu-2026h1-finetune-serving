from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

clf = pipeline("text-classification", model="out/feedback/model")
app = FastAPI(title="hs-feedback-classifier")

class Req(BaseModel):
    texts: list[str]

@app.post("/classify")
def classify(req: Req):
    return [{"text": t, **r} for t, r in zip(req.texts, clf(req.texts[:32], truncation=True))]

# 실행: uvicorn serve:app --port 8000

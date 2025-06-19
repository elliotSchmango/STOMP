from fastapi import FastAPI
import random

app = FastAPI()

@app.get("/predict")
def predict():
    return {"anomaly_score": random.random()}  #use dummy anomaly for now

@app.get("/health")
def health():
    return {"status": "ok"}
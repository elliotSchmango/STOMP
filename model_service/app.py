from fastapi import FastAPI, Request
from model import AnomalyDetector
import random

app = FastAPI()
detector = AnomalyDetector()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    sensors = data.get("sensors")
    if not sensors:
        return {"error": "Missing 'sensors' field"}
    score = detector.predict(sensors)
    return {"anomaly_score": score}
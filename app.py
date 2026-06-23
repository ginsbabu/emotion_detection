from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

import torch
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

MODEL_DIR = "./emotion5_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()

# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(
    title="Emotion Detection API",
    version="1.0"
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# ==========================================
# REQUEST SCHEMA
# ==========================================

class TextRequest(BaseModel):
    text: str

# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_emotion(text):

    encoded = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoded)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    predicted_id = torch.argmax(
        probabilities,
        dim=-1
    ).item()

    confidence = (
        probabilities[0][predicted_id].item()
        * 100
    )

    emotion = model.config.id2label[
        predicted_id
    ]

    return {
        "text": text,
        "emotion": emotion,
        "confidence": round(confidence, 2)
    }

# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/")
def home():
    return {
        "message": "Emotion Detection API Running"
    }

@app.post("/predict")
def predict(request: TextRequest):

    result = predict_emotion(
        request.text
    )

    return result

if __name__== "__main__":
    uvicorn.run("appapi:app",host="0.0.0.0",port=8000,reload="True")
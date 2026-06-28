from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect, LangDetectException

# ==========================================
# MODEL DIRECTORIES
# ==========================================

HINDI_MODEL_DIR = "./hindi_emotion_model1"
ENGLISH_MODEL_DIR = "./english_emotion_model"

# ==========================================
# LOAD BOTH MODELS
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model

print("Loading Hindi model...")
hindi_tokenizer, hindi_model = load_model(HINDI_MODEL_DIR)

print("Loading English model...")
english_tokenizer, english_model = load_model(ENGLISH_MODEL_DIR)

print("Both models ready.")

# ==========================================
# LANGUAGE DETECTION
# ==========================================

def detect_language(text: str) -> str:
    """
    Returns 'hi' for Hindi, 'en' for English.
    Falls back to 'en' if detection fails or
    language is neither Hindi nor English.
    """
    try:
        lang = detect(text)
        if lang == "hi":
            return "hi"
        return "en"          # treat all other languages as English
    except LangDetectException:
        return "en"          # fallback on very short / ambiguous text

# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(title="Emotion Detection API", version="2.0")

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

def predict_emotion(text: str) -> dict:

    # 1. Detect language → pick the right model
    lang = detect_language(text)

    if lang == "hi":
        tokenizer = hindi_tokenizer
        model     = hindi_model
    else:
        tokenizer = english_tokenizer
        model     = english_model

    # 2. Tokenize & run inference
    encoded = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoded)

    probabilities = torch.softmax(outputs.logits, dim=-1)
    predicted_id  = torch.argmax(probabilities, dim=-1).item()
    confidence    = probabilities[0][predicted_id].item() * 100
    emotion       = model.config.id2label[predicted_id]

    return {
        "text":       text,
        "language":   lang,           # 'hi' or 'en' — useful for debugging
        "model_used": "emotion4_model" if lang == "hi" else "emotion5_model",
        "emotion":    emotion,
        "confidence": round(confidence, 2),
    }

# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/")
def home():
    return {"message": "Multilingual Emotion Detection API Running"}

@app.post("/predict")
def predict(request: TextRequest):
    return predict_emotion(request.text)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
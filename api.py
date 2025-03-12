import joblib
import torch
from transformers import BertTokenizer, BertModel
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# Load Tokenizer and Model
print("Loading BERT Tokenizer and Model...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Load trained models
print("Loading trained models...")
logistic_model = joblib.load("Models/logistic_model.pkl")
rf_model = joblib.load("Models/random_forest_model.pkl")
xgb_model = joblib.load("Models/xgboost_model.pkl")

# Initialize FastAPI
app = FastAPI(title="Phishing Email Classifier API", description="Detects phishing emails using BERT embeddings.")

# Request Schema
class EmailInput(BaseModel):
    text: str
    model: str  # Choose from 'logistic_regression', 'random_forest', 'xgboost', or 'all'

# Function to Generate BERT Embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding.reshape(1, -1)  # Reshape for model input

# Function to get probabilities safely
def safe_predict_proba(model, embedding):
    """Returns phishing probability if available, otherwise None."""
    return float(model.predict_proba(embedding)[0][1]) if hasattr(model, "predict_proba") else None

# Root Endpoint
@app.get("/")
def home():
    return {"message": "Phishing Email Classifier API is running!"}

# Prediction Route
@app.post("/predict/")
def predict_email(email: EmailInput):
    # Generate BERT embedding
    embedding = get_bert_embedding(email.text)

    # If a specific model is selected
    if email.model == "all": # replace by logistic_regression
        clf = logistic_model
    elif email.model == "random_forest":
        clf = rf_model
    elif email.model == "xgboost":
        clf = xgb_model
    elif email.model == "all":
        # Get predictions and probabilities from all models
        logit_pred = logistic_model.predict(embedding)[0]
        logit_prob = safe_predict_proba(logistic_model, embedding)

        rf_pred = rf_model.predict(embedding)[0]
        rf_prob = safe_predict_proba(rf_model, embedding)

        xgb_pred = xgb_model.predict(embedding)[0]
        xgb_prob = safe_predict_proba(xgb_model, embedding)

        # Majority Voting: 2 out of 3 must agree
        final_pred = 1 if sum([logit_pred, rf_pred, xgb_pred]) >= 2 else 0

        # Average confidence score (only for models with valid probabilities)
        valid_probs = [p for p in [logit_prob, rf_prob, xgb_prob] if p is not None]
        final_confidence = float(np.mean(valid_probs)) if valid_probs else 0.5

        return {
            "prediction": "phishing" if final_pred == 1 else "not phishing",
            "confidence_score": round(final_confidence, 4),
            "model_predictions": {
                "logistic_regression": {
                    "prediction": "phishing" if logit_pred == 1 else "not phishing",
                    "confidence_score": round(logit_prob, 4) if logit_prob is not None else "N/A"
                },
                "random_forest": {
                    "prediction": "phishing" if rf_pred == 1 else "not phishing",
                    "confidence_score": round(rf_prob, 4) if rf_prob is not None else "N/A"
                },
                "xgboost": {
                    "prediction": "phishing" if xgb_pred == 1 else "not phishing",
                    "confidence_score": round(xgb_prob, 4) if xgb_prob is not None else "N/A"
                },
            }
        }
    else:
        return {"error": "Invalid model name. Choose from 'logistic_regression', 'random_forest', 'xgboost', or 'all'."}

    # Predict for a single model selection
    prediction = clf.predict(embedding)[0]
    confidence = safe_predict_proba(clf, embedding)

    return {
        "prediction": "phishing" if prediction == 1 else "not phishing",
        "confidence_score": round(float(confidence), 4) if confidence is not None else "N/A"
    }

# Startup Message
print("\n🚀 API is ready! Run it with: uvicorn api:app --reload\n")
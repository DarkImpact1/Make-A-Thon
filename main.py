import joblib
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# Load the trained models using joblib
logistic_model = joblib.load("logistic_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")

# Load BERT Tokenizer and Model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

tokenizer.save_pretrained("bert_tokenizer/")
model.save_pretrained("bert_model/")

bert_tokenizer = BertTokenizer.from_pretrained("bert_tokenizer/")
bert_model = BertModel.from_pretrained("bert_model/")

# Function to generate BERT embeddings for input text
def get_bert_embedding(text):
    inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
    
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding.reshape(1, -1)  # Reshape to match model input

# Function to predict phishing or not phishing
def predict_phishing(text):
    embedding = get_bert_embedding(text)

    # Predict using all three models
    logit_pred = logistic_model.predict(embedding)[0]
    logit_prob = logistic_model.predict_proba(embedding)[0][1]

    rf_pred = rf_model.predict(embedding)[0]
    rf_prob = rf_model.predict_proba(embedding)[0][1]

    xgb_pred = xgb_model.predict(embedding)[0]
    xgb_prob = xgb_model.predict_proba(embedding)[0][1]

    # Majority voting for final prediction
    final_pred = round((logit_pred + rf_pred + xgb_pred) / 3)
    final_confidence = np.mean([logit_prob, rf_prob, xgb_prob])

    return {
        "Prediction": "Phishing" if final_pred == 1 else "Not Phishing",
        "Confidence": round(final_confidence * 100, 2),
        # "Model Predictions": {
        #     "Logistic Regression": {"Prediction": logit_pred, "Confidence": round(logit_prob * 100, 2)},
        #     "Random Forest": {"Prediction": rf_pred, "Confidence": round(rf_prob * 100, 2)},
        #     "XGBoost": {"Prediction": xgb_pred, "Confidence": round(xgb_prob * 100, 2)},
        # }
    }

# Example usage
if __name__ == "__main__":
    sample_text = input("Enter email text: ")
    result = predict_phishing(sample_text)
    print("\n🔍 Prediction Result:")
    print(result)

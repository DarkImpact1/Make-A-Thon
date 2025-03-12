import joblib
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import streamlit as st

# Load Models
@st.cache_resource
def load_models():
    logistic_model = joblib.load("Models/logistic_model.pkl")
    rf_model = joblib.load("Models/random_forest_model.pkl")
    xgb_model = joblib.load("Models/xgboost_model.pkl")
    return logistic_model, rf_model, xgb_model

# Load BERT Tokenizer & Model
@st.cache_resource
def load_bert():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    return tokenizer, model

# Load everything
logistic_model, rf_model, xgb_model = load_models()
bert_tokenizer, bert_model = load_bert()

# Function to get BERT embeddings
def get_bert_embedding(text):
    inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy().reshape(1, -1)

# Function to predict phishing
def predict_phishing(text):
    embedding = get_bert_embedding(text)

    logit_pred = logistic_model.predict(embedding)[0]
    logit_prob = logistic_model.predict_proba(embedding)[0][1]

    rf_pred = rf_model.predict(embedding)[0]
    rf_prob = rf_model.predict_proba(embedding)[0][1]

    xgb_pred = xgb_model.predict(embedding)[0]
    xgb_prob = xgb_model.predict_proba(embedding)[0][1]

    # Majority voting
    final_pred = round((logit_pred + rf_pred + xgb_pred) / 3)
    final_confidence = np.mean([logit_prob, rf_prob, xgb_prob])

    return {
        "Prediction": "Phishing" if final_pred == 1 else "Not Phishing",
        "Confidence": round(final_confidence * 100, 2),
        "Model Predictions": {
            "Logistic Regression": {"Prediction": "Phishing" if logit_pred == 1 else "Not Phishing", "Confidence": round(logit_prob * 100, 2)},
            "Random Forest": {"Prediction": "Phishing" if rf_pred == 1 else "Not Phishing", "Confidence": round(rf_prob * 100, 2)},
            "XGBoost": {"Prediction": "Phishing" if xgb_pred == 1 else "Not Phishing", "Confidence": round(xgb_prob * 100, 2)},
        }
    }

# Streamlit UI
st.title("🔍 Phishing Email Detector")
st.markdown("Enter an email text below, and the model will determine if it's **Phishing** or **Not Phishing**.")

email_text = st.text_area("✉️ Paste your email text here:", "")

if st.button("🚀 Detect Phishing"):
    if email_text.strip():
        result = predict_phishing(email_text)
        st.subheader("✅ Prediction Result:")
        st.write(f"**Prediction:** {result['Prediction']}")
        st.write(f"**Confidence Score:** {result['Confidence']}%")

        with st.expander("📊 Model-wise Predictions"):
            for model_name, details in result["Model Predictions"].items():
                st.write(f"🔹 **{model_name}** - {details['Prediction']} ({details['Confidence']}%)")

    else:
        st.warning("⚠️ Please enter some text to analyze!")

st.markdown("---")
st.caption("👨‍💻 Developed by Team Techno Titans | Powered by BERT & ML Models")

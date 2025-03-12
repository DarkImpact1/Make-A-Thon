import joblib

xgb = joblib.load("Models/xgboost_model.pkl")
print(f"loaded model ")
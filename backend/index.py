from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
from preprocess import preprocess_data

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_methods=["*"],
  allow_headers=["*"],
)

model_dir = "./models/"
try:
  model = joblib.load(f"{model_dir}best_model.pkl")
  scaler = joblib.load(f"{model_dir}scaler.pkl")
  feature_columns = joblib.load(f"{model_dir}feature_columns.pkl")
  model_metadata = joblib.load(f"{model_dir}model_metadata.pkl")
  print(f" Successfully loaded: {model_metadata.get('model_name', 'Unknown Model')}")
except Exception as e:
  print(f"Error loading model files from {model_dir}: {e}")

EDUCATION_MAP = {
  1: "Preschool", 2: "1st-4th", 3: "5th-6th", 4: "7th-8th",
  5: "9th", 6: "10th", 7: "11th", 8: "12th", 9: "HS-grad",
  10: "Some-college", 11: "Assoc-voc", 12: "Assoc-acdm",
  13: "Bachelors", 14: "Masters", 15: "Prof-school", 16: "Doctorate"
}


class InputData(BaseModel):
  age: int
  sex: str
  education_num: int
  hours_per_week: int
  capital_gain: int
  capital_loss: int
  workclass: str
  marital_status: str
  occupation: str
  relationship: str


@app.get("/")
def root():
  return {"message": "Income Prediction API is running!", "online": True}


@app.get("/model-info")
def model_info():
  return model_metadata


@app.post("/predict")
def predict(data: InputData):
  education_label = EDUCATION_MAP.get(data.education_num, "Bachelors")
  input_dict = {
    "age": [data.age],
    "sex": [data.sex],
    "education-num": [data.education_num],
    "hours-per-week": [data.hours_per_week],
    "capital-gain": [data.capital_gain],
    "capital-loss": [data.capital_loss],
    "workclass": [data.workclass],
    "marital-status": [data.marital_status],
    "occupation": [data.occupation],
    "relationship": [data.relationship],
    "fnlwgt": [200000], 
    "education": [education_label],
    "race": ["White"],
    "native-country": ["United-States"],
  }
  df = pd.DataFrame(input_dict)
  df_processed = preprocess_data(df)

  if 'sex' in df_processed.columns:
    df_processed['sex'] = df_processed['sex'].map({'Male': 1, 'Female': 0}).fillna(0)
  final_df = pd.DataFrame(columns=feature_columns)
  for col in feature_columns:
    if col in df_processed.columns:
      final_df[col] = df_processed[col]
    else:
      final_df[col] = 0  
  final_df = final_df.fillna(0)[feature_columns]
  df_scaled = scaler.transform(final_df)
  probability = model.predict_proba(df_scaled)[0]
  prediction_result = 1 if probability[1] > 0.4 else 0
  result_text = ">50K" if prediction_result == 1 else "<=50K"
  confidence = round(float(max(probability)) * 100, 2)
  return {
    "prediction": result_text,
    "confidence": confidence,
    "model_name": model_metadata.get("model_name", "XGBoost"),
    "f1": model_metadata.get("f1", "N/A"),
    "details": {
      "prob_less_50k": round(float(probability[0]) * 100, 2),
      "prob_more_50k": round(float(probability[1]) * 100, 2),
    }
  }


app.mount("/analytics", StaticFiles(directory="analytics"), name="analytics")
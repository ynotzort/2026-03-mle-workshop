import pickle
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field
from loguru import logger

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_PATH: str = "./model.bin"
    
settings = Settings()

with open(settings.MODEL_PATH, "rb") as f_in:
    model = pickle.load(f_in)
logger.info(f"Model loaded from : {settings.MODEL_PATH}")

# "feature engineering" part
def prepare_features(raw_features: dict[str, Any]) -> dict[str,Any]:
    features = {}
    features["PULocationID"] = str(raw_features["PULocationID"])
    features["DOLocationID"] = str(raw_features["DOLocationID"])
    features["trip_distance"] = float(raw_features["trip_distance"])
    return features

def predict(features: dict[str, Any]) -> float:
    preds = model.predict(features)
    return float(preds[0])

# maybe convert minutes to seconds
def postprocess(prediction: float) -> float:
    return prediction

### server
class PredictRequest(BaseModel):
    """Inputs required for prediction
    """
    PULocationID: str = Field(description="PickUp location ID")
    DOLocationID: str = Field(description="DropOff location ID")
    trip_distance: float = Field(description="Distance in km")

app = FastAPI()

@app.post("/predict")
def predict_endpoint(predict_request: PredictRequest) -> dict[str, Any]:
    prepared_features = prepare_features(predict_request.model_dump())
    prediction = predict(prepared_features)
    final_prediction = postprocess(prediction)
    return dict(prediction=dict(duration=final_prediction))
    


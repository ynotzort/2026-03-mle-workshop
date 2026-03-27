import pickle
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

with open("./models/2022-01.bin", "rb") as f_in:
    model = pickle.load(f_in)
    
# trip = {
#     "PULocationID": "43",
#     "DOLocationID": "238",
#     "trip_distance": 1.16,
# }

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
    PULocationID: str
    DOLocationID: str
    trip_distance: float

app = FastAPI()

@app.post("/predict")
def predict_endpoint(predict_request: PredictRequest) -> dict[str, Any]:
    prepared_features = prepare_features(predict_request.model_dump())
    prediction = predict(prepared_features)
    final_prediction = postprocess(prediction)
    return dict(prediction=dict(duration=final_prediction))
    


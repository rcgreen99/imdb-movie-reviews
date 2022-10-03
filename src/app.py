from fastapi import FastAPI
from pydantic import BaseModel
from src.predictor import Predictor


class Review(BaseModel):
    text: str


MODEL_PATH = "models/model-93acc.pth"


app = FastAPI()

predictor = Predictor(MODEL_PATH)


@app.post("/predict")
async def predict(review: Review):
    prediction = predictor.predict(review.text)
    return {"prediction": prediction}

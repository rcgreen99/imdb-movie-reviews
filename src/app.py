from fastapi import FastAPI, Form, Request
from src.predictor import Predictor
from fastapi.templating import Jinja2Templates


app = FastAPI()

templates = Jinja2Templates(directory="templates")

MODEL_PATH = "models/model-93acc.pth"
predictor = Predictor(MODEL_PATH)


@app.get("/")
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(review: str = Form()):
    prediction = predictor.predict(review)
    return {"prediction": prediction}

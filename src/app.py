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


@app.post("/prediction")
async def predict(request: Request, review: str = Form()):
    ouput_val, prediction = predictor.predict(review)
    if prediction:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    return templates.TemplateResponse(
        "prediction.html",
        {"request": request, "sentiment": sentiment, "output_val": ouput_val},
    )

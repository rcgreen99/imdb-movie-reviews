from fastapi import FastAPI

# create fats api endpoint called app
app = FastAPI()

model = None


@app.get("/")
def read_root():
    return {"Hello": "World"}

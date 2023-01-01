from fastapi import FastAPI
from pydantic import BaseModel

from predict import get_prediction

class Text(BaseModel):
    txt: str

app = FastAPI()

@app.get("/")
def home():
    return {"Data": "Welcome Home"}

@app.post("/predict")
def predict(text: Text):
    if(len(text.txt)==0):
        return {"Error": "Please! Enter Text"}

    response = get_prediction(text.txt)

    if(response[0]==-1):
        return {"Error": response[1]}

    pred = response[1]
    sentiment = "negative"
    if pred>0.5:
        sentiment = "positive"
    return {"Text": text.txt, "positive": round(pred, 2), "negative": round(1-pred, 2), "sentiment": sentiment}
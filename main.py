from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prediction import make_prediction

app = FastAPI()

# Разрешаем CORS для фронтенда (index.html)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    input: dict

@app.post("/api/predict")
async def predict(data: PredictionRequest):
    result = make_prediction(data.input)
    return result

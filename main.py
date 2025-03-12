import pickle

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("composer.pkl", "rb") as f:
    preprocessor = pickle.load(f)

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]

indexator = 1

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    global indexator
    dictionary = item.model_dump()
    df = pd.DataFrame(dictionary, index=[indexator])
    df = df.drop('selling_price', axis=1)
    indexator += 1
    df = preprocessor.transform(df)
    return model.predict(df)

@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    output = []
    for item in items:
        output.append(predict_item(item))
    return output

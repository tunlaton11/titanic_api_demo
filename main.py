from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from model.model_manager import load_model
from model.model_utility import preprocess_query
import pandas as pd
from typing import Union

model = load_model()
app = FastAPI()

class Passenger(BaseModel):
    PassengerId: int
    Pclass: int
    Name: str
    Sex: str
    Age: int
    SibSp: int
    Parch: int
    Ticket: str
    Fare: float
    Cabin: Optional[str] = None
    Embarked: str


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/predict")
async def predict_survived(passengerId:int,
                            pclass:int,
                            name:str,
                            sex:str,
                            age:int,
                            sibSp:int,
                            parch:int,
                            ticket:str,
                            fare:float,
                            embarked:str,
                            cabin: Union[str, None] = None):
    query = [[passengerId, pclass, name, sex, age, sibSp, parch, ticket, fare, cabin, embarked]]
    print('Query:')
    print(query)
    query_df = pd.DataFrame(query,
                            columns=['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'])
    #print(query_df)
    new_query_df = preprocess_query(query_df)
    #print(new_query_df)
    prediction = model.predict(new_query_df)
    print("Prediction result: " + str(prediction))
    return {"Survived": int(prediction)}

@app.post("/predict_valid")
async def predict_survived(passenger: Passenger):
    query_dict = passenger.dict()
    #print(query_dict)
    query_df = pd.json_normalize(query_dict)
    #print(query_df)
    new_query_df = preprocess_query(query_df)
    #print(new_query_df)
    prediction = model.predict(new_query_df)
    print("Prediction result: " + str(prediction))
    return {"Survived": int(prediction)}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
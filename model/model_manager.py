from model.model_utility import load_deployable_model, preprocess_query
import json
import pandas as pd
import os

def load_model():
    op_dir = '/trained_model/'
    model_file = 'trained_model.pkl'

    model = load_deployable_model(os.getcwd() + op_dir + model_file)
    return model

def main():
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)
    # print(os.getcwd())

    model = load_model()

    json_item = {
        "PassengerId": 248,
        "Pclass": 2,
        "Name": "Hamalainen, Mrs. William (Anna)",
        "Sex": "female",
        "Age": 24,
        "SibSp": 0,
        "Parch": 2,
        "Ticket": "250649",
        "Fare": 14.5000,
        "Cabin": "nocabin",
        "Embarked": "S"
    }

    # json string like one sent by request
    json_string = json.dumps(json_item)
    # print(type(json_string))
    # print(json_string)
    # print('--------------------')
    query_dict = json.loads(json_string)
    # print(query_dict)
    query_df = pd.json_normalize(query_dict)
    # print(query_df)

    new_query_df = preprocess_query(query_df)
    # print(new_query_df)
    prediction = model.predict(new_query_df)
    print("Prediction result: " + str(prediction))


# this means that if this script is executed, then
# main() will be executed
if __name__ == '__main__':
    main()
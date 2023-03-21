
import pandas as pd
import dill
import json
import os


def predict():
    path = os.environ.get('PROJECT_PATH')
    name = os.listdir(f'{path}/data/models')
    with open(f'{path}/data/models/{name[0]}', 'rb') as file:
        model = dill.load(file)
    df_pred = pd.DataFrame(columns=['car_id', 'pred'])
    for filename in os.listdir(f'{path}/data/test'):
        with open(os.path.join(f'{path}/data/test', filename), 'r') as j:
            data = json.load(j)
            df = pd.DataFrame([data])
            pred = model.predict(df)
            x = {'car_id': df.id, 'pred': pred}
            df1 = pd.DataFrame(x)
            df_pred = pd.concat([df_pred, df1], axis=0)
    df_pred.to_csv(f'{path}/data/predictions/pred.csv')



if __name__ == '__main__':
    predict()

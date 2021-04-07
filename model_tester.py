import json
import requests
import pandas as pd

from settings.constants import TRAIN_CSV, TEST_CSV
from utils.Data_preprocessor import DataLoader
#from Model_fitter import BEST_SCORE

train_set = pd.read_csv(TRAIN_CSV, header=0)
test_set = pd.read_csv(TEST_CSV, header=0)

train_x, train_y = train_set, train_set['Survived']
test_x = test_set

dataloader = DataLoader()
dataloader.fit(train_x)
train_x = dataloader.load_data()
dataloader.fit(test_x)
test_x = dataloader.load_data()

req_data = {'data': json.dumps(test_x.to_dict())}
response = requests.get('http://127.0.0.1:8000/predict', data=req_data)
api_predict = response.json()['prediction']
# print(req_data)
# print("accuracy_score:", accuracy_score(api_predict, y_valid))
# print("roc_auc_score:", roc_auc_score(api_predict, y_valid))
print('predict: ', api_predict[:10])
# print('Best cross val score:', BEST_SCORE)


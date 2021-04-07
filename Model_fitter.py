import pickle
import pandas as pd

from utils.Data_preprocessor import DataLoader
from settings.constants import TRAIN_CSV
from utils.GridSearchPipeline import GrSPipeLine

train = pd.read_csv(TRAIN_CSV)

X = train
y = train['Survived']

dataloader = DataLoader()
dataloader.fit(X)
X = dataloader.load_data()

pipeline = GrSPipeLine(X, y)
best_estimator, best_score = pipeline.grid_search_estimator()

with open('models/GridSearch.pickle', 'wb')as f:
    pickle.dump(best_estimator.fit(X, y), f)

BEST_SCORE = best_score
print(X)
print('Best cross val score:', BEST_SCORE)

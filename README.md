# Titanic_Disaster_ML_Model
First ML model

Here you can find my first ML project. It's Titanic Disaster classification task.
The data were taken from Kaggle: https://www.kaggle.com/c/titanic/data

The main goal of the project is to predict whether person will alive in Titanic given surrounding circumstances.

The project was done such as data was cleaned. After that complicated Pipelines inside several GridSearches were built and was found the best way to preprocess data
and best model. A simple FLask API was built as well, that allows to predict new observations through API request.

Original Dataset contains 10 columns, one of them is Survived column - our target column.

The project structure contains:
- Data folder with train and test data in CSV format
- Settings folder with all needed constants
- Models folder, which contain saved estimators, that can be used for predictions
- Utils folder, which contains all files needed for preprocessing and modeling dataset
- Model_fitter.py file: fit model on the train data and save fitted estimator as pickle file to Models folder
- app.py file: produce simple Flask API
- model_tester.py file: predict test data through Flask API
- titanic-notebook.ipynb file: Jupyter notebook with the flow of project. Contain data exploration, EDA, data preparation, feature engineering and modeling.
- Dockerfile: for push this project into Docker Image

In order to see my thoughts during this project take a look on Jupyter notebook.

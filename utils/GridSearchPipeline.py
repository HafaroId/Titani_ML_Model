from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, train_test_split


class GrSPipeLine:
    """
    A class used to construct Pipeline inside GridSearch

    ...

    Attributes
    __________
    X : pandas DataFrame
        train dataset
    y : pandas DataFrame
        target dataset

    Methods
    -------
    pipeline_constructor()
        construct Pipelines for numerical and categorical columns
    """
    def __init__(self, X, y):
        self.X = X.copy()
        self.y = y.copy()
        self.preprocessor = None
        self.preprocessor_dict = None
        self.pipeline_constructor()

    def pipeline_constructor(self):
        """
        Construct Pipelines for numerical and categorical columns.
        Create self.preprocessor and self.preprocessor_dict for final Pipeline and further GridSearch
        """

        # At first we'll find columns with numerical and categorical values
        num_features = [col for col in self.X.columns if self.X[col].dtype in ['int64', 'float64']]
        cat_features = [col for col in self.X.columns if self.X[col].dtype == 'object']

        # Pipeline for numerical features that contain imputing, scaling and normalization operations
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer()),
            ('scaler', 'passthrough'),
            ('norm', 'passthrough')
        ])

        # Pipeline for categorical features that contain imputing and one-hot encoding operations
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combining of num and cat Pipelines into one preprocessor step using ColumnTransformer.
        # This preprocessor will be used in final Pipeline and further in GridSearch
        self.preprocessor = ColumnTransformer(transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ])

        # Creating of dictionary with preprocessing parameters. This dict will be used in GridSearch
        self.preprocessor_dict = dict(preprocessor__num__imputer__strategy=['mean', 'median'],
                                      preprocessor__num__scaler=[StandardScaler(), RobustScaler(),
                                                                 RobustScaler(with_centering=False), MinMaxScaler()],
                                      preprocessor__num__norm=[PowerTransformer()])

    def RFC_grid_search(self, grid_params={}):
        """
        Method create classifier using preprocessor and RandomForestClassifier and fit this Pipeline to X, y

        Parameters
        ----------
        grid_params : empty dict
            Used for constructing final param_dict in GridSearch

        Returns
        -------
        grid_search.best_score_
            best score of GridSearch

        grid_search.best_estimator_
            best estimator of GridSearch
        """
        RFC_classifier = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier())
        ])
        RFC_dict = dict(classifier__n_estimators=[100, 500, 900],  # classifier hyperparameters dict
                        classifier__min_samples_leaf=[1, 2, 4])

        grid_params.update(self.preprocessor_dict)
        grid_params.update(RFC_dict) # dict used in GridSearch as param_grid

        grid_search = GridSearchCV(RFC_classifier, param_grid=grid_params, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(self.X, self.y)

        return grid_search.best_score_, grid_search.best_estimator_

    def SVC_grid_search(self, grid_params={}):
        """
        Method create classifier using preprocessor and SVC and fit this Pipeline to X, y

        Parameters
        ----------
        grid_params : empty dict
            Used for constructing final param_dict in GridSearch

        Returns
        -------
        grid_search.best_score_
            best score of GridSearch

        grid_search.best_estimator_
            best estimator of GridSearch
        """
        SVC_classifier = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', SVC(probability=True))
        ])
        SVC_dict = dict(classifier__kernel=['rbf', 'linear', 'sigmoid'],  # classifier hyperparameters dict
                        classifier__C=[0.001, 0.01, 0.1, 1])
        grid_params.update(self.preprocessor_dict)
        grid_params.update(SVC_dict)

        grid_search = GridSearchCV(SVC_classifier, param_grid=grid_params, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(self.X, self.y)
        return grid_search.best_score_, grid_search.best_estimator_

    def SGD_grid_search(self, grid_params={}):
        """
        Method create classifier using preprocessor and SGD and fit this Pipeline to X, y

        Parameters
        ----------
        grid_params : empty dict
            Used for constructing final param_dict in GridSearch

        Returns
        -------
        grid_search.best_score_
            best score of GridSearch

        grid_search.best_estimator_
            best estimator of GridSearch
        """
        SGD_classifier = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', SGDClassifier(early_stopping=True))
        ])
        SGD_dict = dict(classifier__alpha=[0.001, 0.01, 0.1, 1])  # classifier hyperparameters dict
        grid_params.update(self.preprocessor_dict)
        grid_params.update(SGD_dict)

        grid_search = GridSearchCV(SGD_classifier, param_grid=grid_params, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(self.X, self.y)
        return grid_search.best_score_, grid_search.best_estimator_

    def grid_search_estimator(self, estimators_dict={}):
        """
        Method choose best Pipeline basing on best_score

        Parameters
        ----------
        grid_params : empty dict

        Returns
        -------
        best_estimator
            best estimator of RFC, SVC, SGD based GridSearch Pipelines
        """
        rfc_score, rfc_best_estimator = self.RFC_grid_search()
        svc_score, svc_best_estimator = self.SVC_grid_search()
        sgd_score, sgd_best_estimator = self.SGD_grid_search()

        estimators_dict['RFC'] = [rfc_score, rfc_best_estimator]
        estimators_dict['SV'] = [svc_score, svc_best_estimator]
        estimators_dict['SGD'] = [sgd_score, sgd_best_estimator]
        score_list = [i[0] for i in list(estimators_dict.values())]

        f = 0
        max_idx = 0
        for j in range(len(score_list)):
            if score_list[j] > f:
                f = score_list[j]
                max_idx = j

        #        max_idx = list.index(max(estimators_list))
        print('best_estimator:', list(estimators_dict.keys())[max_idx],
              '\nbest_score:', list(estimators_dict.values())[max_idx][0])
        if max_idx == 0:
            return rfc_best_estimator, rfc_score
        elif max_idx == 1:
            return svc_best_estimator, svc_score
        else:
            return sgd_best_estimator, sgd_score

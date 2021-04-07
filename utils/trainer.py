from sklearn.svm import SVR

class Estimator:
    @staticmethod
    def fit(train_x, train_y):
        return SVR(epsilon=0.2).fit(train_x, train_y)

    @staticmethod
    def predict(trained, test_x):
        return trained.predict(test_x)
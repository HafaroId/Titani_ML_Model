class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    def load_data(self):
        self.dataset['Title'] = self.dataset.Name.str.split(' ').str[1]
        self.dataset['Cabin_class'] = self.dataset.Cabin.str[0]
        self.dataset['Fare'] = round(self.dataset['Fare'])
        self.dataset = self.dataset.drop(['PassengerId', 'Ticket', 'Name', 'Cabin'], axis=1)
        if 'Survived' in self.dataset.columns:
            self.dataset = self.dataset.drop(['Survived'], axis=1)
        return self.dataset

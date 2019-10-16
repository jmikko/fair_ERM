from load_data import load_adult
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
from measures import equalized_odds_measure_TP
from sklearn.model_selection import GridSearchCV
from collections import namedtuple


class Linear_FERM:
    # The linear FERM algorithm
    def __init__(self, dataset, model, sensible_feature):
        self.dataset = dataset
        self.values_of_sensible_feature = list(set(dataset.data[:, sensible_feature]))
        self.list_of_sensible_feature_train = dataset.data[:, sensible_feature]
        self.val0 = np.min(self.values_of_sensible_feature)
        self.val1 = np.max(self.values_of_sensible_feature)
        self.model = model
        self.sensible_feature = sensible_feature
        self.u = None
        self.max_i = None

    def new_representation(self, examples):
        if self.u is None:
            tmp = [ex for idx, ex in enumerate(self.dataset.data)
                   if self.dataset.target[idx] == 1 and ex[self.sensible_feature] == self.val1]
            average_A_1 = np.mean(tmp, 0)
            tmp = [ex for idx, ex in enumerate(self.dataset.data)
                   if self.dataset.target[idx] == 1 and ex[self.sensible_feature] == self.val0]
            average_not_A_1 = np.mean(tmp, 0)
            self.u = -(average_A_1 - average_not_A_1)
            self.max_i = np.argmax(self.u)

        new_examples = np.array([ex - self.u * (ex[self.max_i] / self.u[self.max_i]) for ex in examples])
        new_examples = np.delete(new_examples, self.max_i, 1)
        return new_examples

    def predict(self, examples):
        if self.u is None:
            print('Model not trained yet!')
            return 0

        new_examples = np.array([ex - self.u * (ex[self.max_i] / self.u[self.max_i]) for ex in examples])
        new_examples = np.delete(new_examples, self.max_i, 1)
        prediction = self.model.predict(new_examples)
        return prediction

    def fit(self):
        # Evaluation of the empirical averages among the groups
        tmp = [ex for idx, ex in enumerate(self.dataset.data)
               if self.dataset.target[idx] == 1 and ex[self.sensible_feature] == self.val1]
        average_A_1 = np.mean(tmp, 0)
        tmp = [ex for idx, ex in enumerate(self.dataset.data)
               if self.dataset.target[idx] == 1 and ex[self.sensible_feature] == self.val0]
        average_not_A_1 = np.mean(tmp, 0)

        # Evaluation of the vector u (difference among the two averages)
        self.u = -(average_A_1 - average_not_A_1)

        # Application of the new representation
        self.max_i = np.argmax(self.u)

        newdata = np.array([ex - self.u * (ex[self.max_i] / self.u[self.max_i]) for ex in self.dataset.data])
        newdata = np.delete(newdata, self.max_i, 1)

        self.dataset = namedtuple('_', 'data, target')(newdata, self.dataset.target)

        # Fitting the linear model by using the new data
        if self.model:
            self.model.fit(self.dataset.data, self.dataset.target)


if __name__ == "__main__":
    # Load Adult dataset
    dataset_train, dataset_test = load_adult(smaller=False)
    sensible_feature = 9  # GENDER
    sensible_feature_values = sorted(list(set(dataset_train.data[:, sensible_feature])))
    print('Different values of the sensible feature', sensible_feature, ':', sensible_feature_values)
    ntrain = len(dataset_train.target)

    # Standard SVM -  Train an SVM using the training set
    print('Grid search...')
    param_grid = [{'C': [0.01, 0.1, 1.0], 'kernel': ['linear']}]
    svc = svm.SVC()
    clf = GridSearchCV(svc, param_grid, n_jobs=1)
    clf.fit(dataset_train.data, dataset_train.target)
    print('Best Estimator:', clf.best_estimator_)

    # Accuracy and Fairness
    pred = clf.predict(dataset_test.data)
    pred_train = clf.predict(dataset_train.data)
    print('Accuracy test:', accuracy_score(dataset_test.target, pred))
    print('Accuracy train:', accuracy_score(dataset_train.target, pred_train))
    # Fairness measure
    EO_train = equalized_odds_measure_TP(dataset_train, clf, [sensible_feature], ylabel=1)
    EO_test = equalized_odds_measure_TP(dataset_test, clf, [sensible_feature], ylabel=1)
    print('DEO test:', np.abs(EO_test[sensible_feature][sensible_feature_values[0]] -
                              EO_test[sensible_feature][sensible_feature_values[1]]))
    print('DEO train:', np.abs(EO_train[sensible_feature][sensible_feature_values[0]] -
                               EO_train[sensible_feature][sensible_feature_values[1]]))

    # Linear FERM
    list_of_sensible_feature_test = dataset_test.data[:, sensible_feature]
    print('\n\n\nGrid search for our method...')
    svc = svm.SVC()
    clf = GridSearchCV(svc, param_grid, n_jobs=1)
    algorithm = Linear_FERM(dataset_train, clf, sensible_feature)
    algorithm.fit()
    print('Best Fair Estimator::', algorithm.model.best_estimator_)

    # Accuracy and Fairness
    pred = algorithm.predict(dataset_test.data)
    pred_train = algorithm.predict(dataset_train.data)
    print('Accuracy test fair:', accuracy_score(dataset_test.target, pred))
    print('Accuracy train fair:', accuracy_score(dataset_train.target, pred_train))
    # Fairness measure
    EO_train = equalized_odds_measure_TP(dataset_train, algorithm, [sensible_feature], ylabel=1)
    EO_test = equalized_odds_measure_TP(dataset_test, algorithm, [sensible_feature], ylabel=1)
    print('DEO test:', np.abs(EO_test[sensible_feature][sensible_feature_values[0]] -
                              EO_test[sensible_feature][sensible_feature_values[1]]))
    print('DEO train:', np.abs(EO_train[sensible_feature][sensible_feature_values[0]] -
                               EO_train[sensible_feature][sensible_feature_values[1]]))

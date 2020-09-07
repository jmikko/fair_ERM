from load_data import load_adult
from sklearn import svm
from sklearn.metrics import accuracy_score
from measures import equalized_odds_measure_TP
from sklearn.model_selection import GridSearchCV
from cvxopt import matrix
import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import rbf_kernel


# Definition of different kernels
def linear_kernel(x1, x2):
    return np.dot(x1, np.transpose(x2))

def gaussian_kernel(x, y, gamma=0.1):
    return np.exp(-gamma * (linalg.norm(x - y)**2))


class FERM(BaseEstimator):
    # FERM algorithm
    def __init__(self, kernel='rbf', C=1.0, sensible_feature=None, gamma=1.0):
        self.kernel = kernel
        self.C = C
        self.fairness = False if sensible_feature is None else True
        self.sensible_feature = sensible_feature
        self.gamma = gamma
        self.w = None

    def fit(self, X, y):
        if self.kernel == 'rbf':
            self.fkernel = lambda x, y: rbf_kernel(x, y, self.gamma)
        elif self.kernel == 'linear':
            self.fkernel = linear_kernel
        else:
            self.fkernel = linear_kernel

        if self.fairness:
            self.values_of_sensible_feature = list(set(self.sensible_feature))
            self.list_of_sensible_feature_train = self.sensible_feature
            self.val0 = np.min(self.values_of_sensible_feature)
            self.val1 = np.max(self.values_of_sensible_feature)
            self.set_A1 = [idx for idx, ex in enumerate(X) if y[idx] == 1
                           and self.sensible_feature[idx] == self.val1]
            self.set_not_A1 = [idx for idx, ex in enumerate(X) if y[idx] == 1
                               and self.sensible_feature[idx] == self.val0]
            # print('self.val0:', self.val0)
            # print('self.val1:', self.val1)
            # print('(y, self.sensible_feature):')
            # for el in zip(y, self.sensible_feature):
            #     print(el)
            self.set_1 = [idx for idx, ex in enumerate(X) if y[idx] == 1]
            self.n_A1 = len(self.set_A1)
            self.n_not_A1 = len(self.set_not_A1)
            self.n_1 = len(self.set_1)

        n_samples, n_features = X.shape

        # Gram matrix
        K = self.fkernel(X, X)

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        # print(y)
        A = cvxopt.matrix(y.astype(np.double), (1, n_samples), 'd')
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # Stack the fairness constraint
        if self.fairness:
            tau = [(np.sum(K[self.set_A1, idx]) / self.n_A1) - (np.sum(K[self.set_not_A1, idx]) / self.n_not_A1)
                   for idx in range(len(y))]
            # print('self.n_A1:', self.n_A1)
            # print('self.n_not_A1:', self.n_not_A1)
            # print('tau:', tau)
            fairness_line = matrix(y * tau, (1, n_samples), 'd')
            A = cvxopt.matrix(np.vstack([A, fairness_line]))
            b = cvxopt.matrix([0.0, 0.0])

        # solve QP problem
        cvxopt.solvers.options['show_progress'] = False
        # print('A:', A)
        # print('Rank(A):', np.linalg.matrix_rank(A))
        # print('Rank([P; A; G])', np.linalg.matrix_rank(np.vstack([P, A, G])))
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-7
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        # print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            XSV = self.fkernel(X, self.sv)
            a_sv_y = np.multiply(self.a, self.sv_y)
            y_predict = [np.sum(np.multiply(np.multiply(self.a, self.sv_y), XSV[i, :])) for i in range(len(X))]

            return y_predict + self.b

    def decision_function(self, X):
        return self.project(X)

    def predict(self, X):
        return np.sign(self.project(X))

    def score(self, X_test, y_test):
        predict = self.predict(X_test)
        acc = accuracy_score(y_test, predict)
        return acc


if __name__ == "__main__":
    # Load Adult dataset (a smaller version!)
    dataset_train, dataset_test = load_adult(smaller=True)
    sensible_feature = 9  # GENDER
    sensible_feature_values = sorted(list(set(dataset_train.data[:, sensible_feature])))
    print('Different values of the sensible feature', sensible_feature, ':', sensible_feature_values)
    ntrain = len(dataset_train.target)

    # Standard SVM - Train an SVM using the training set
    print('Grid search for SVM...')
    grid_search_complete = 1
    if grid_search_complete:
        param_grid = [{'C': [0.1, 1, 10.0],
                       'gamma': [0.1, 0.01],
                       'kernel': ['rbf']}
                      ]
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

    #  FERM algorithm
    print('\n\n\nGrid search for our method...')
    algorithm = FERM(sensible_feature=dataset_train.data[:, sensible_feature])
    clf = GridSearchCV(algorithm, param_grid, n_jobs=1)
    clf.fit(dataset_train.data, dataset_train.target)
    print('Best Fair Estimator:', clf.best_estimator_)

    # Accuracy and Fairness
    y_predict = clf.predict(dataset_test.data)
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

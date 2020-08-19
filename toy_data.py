import matplotlib.pyplot as plt
import numpy as np



def generate_toy_data(n_samples, n_samples_low, n_dimensions):
    np.random.seed(0)
    varA = 0.8
    aveApos = [-1.0] * n_dimensions
    aveAneg = [1.0] * n_dimensions
    varB = 0.5
    aveBpos = [0.5] * int(n_dimensions / 2) + [-0.5] * int(n_dimensions / 2 + n_dimensions % 2)
    aveBneg = [0.5] * n_dimensions

    X = np.random.multivariate_normal(aveApos, np.diag([varA] * n_dimensions), n_samples)
    X = np.vstack([X, np.random.multivariate_normal(aveAneg, np.diag([varA] * n_dimensions), n_samples)])
    X = np.vstack([X, np.random.multivariate_normal(aveBpos, np.diag([varB] * n_dimensions), n_samples_low)])
    X = np.vstack([X, np.random.multivariate_normal(aveBneg, np.diag([varB] * n_dimensions), n_samples)])
    sensible_feature = [1] * (n_samples * 2) + [0] * (n_samples + n_samples_low)
    sensible_feature = np.array(sensible_feature)
    sensible_feature.shape = (len(sensible_feature), 1)
    X = np.hstack([X, sensible_feature])
    y = [1] * n_samples + [0] * n_samples + [1] * n_samples_low + [0] * n_samples
    y = np.array(y)
    sensible_feature_id = len(X[1, :]) - 1
    idx_A = list(range(0, n_samples * 2))
    idx_B = list(range(n_samples * 2, n_samples * 3 + n_samples_low))

    return X, y, sensible_feature_id, idx_A, idx_B


if __name__ == "__main__":
    from linear_ferm import Linear_FERM
    from sklearn import svm
    from sklearn.metrics import accuracy_score
    import numpy as np
    from measures import equalized_odds_measure_TP
    from sklearn.model_selection import GridSearchCV
    from collections import namedtuple

    n_samples = 100 * 20
    n_samples_low = 20 * 20
    X, y, sensible_feature, idx_A, idx_B = generate_toy_data(n_samples=n_samples,
                                                                n_samples_low=n_samples_low,
                                                                n_dimensions=2)
    point_size = 150
    linewidth = 6

    step = 30
    alpha = 0.5
    plt.scatter(X[0:n_samples * 2:step, 0], X[0:n_samples * 2:step, 1], marker='o', s=point_size, c=y[0:n_samples * 2:step], edgecolors='k', label='Group A', alpha=alpha)
    plt.scatter(X[n_samples * 2::step, 0], X[n_samples * 2::step, 1], marker='s', s=point_size, c=y[n_samples * 2::step], edgecolors='k', label='Group B', alpha=alpha)
    plt.legend()
    plt.title("Dataset")
    plt.colorbar()

    sensible_feature_values = sorted(list(set(X[:, sensible_feature])))
    dataXy = namedtuple('_', 'data, target')(X, y)

    C_list = [10**v for v in range(-6, 3)]
    print("List of C tested:", C_list)
    # Standard SVM -  Train an SVM using the training set
    acc_list, deo_list, acc_fair_list, deo_fair_list = [], [], [], []

    print('STANDARD LINEAR SVM')
    for c in C_list:
        print('C:', c)
        clf = svm.SVC(kernel='linear', C=c)
        clf.fit(X, y)

        # Accuracy and Fairness
        pred = clf.predict(X)
        print('Accuracy:', accuracy_score(y, pred))
        # Fairness measure
        EO = equalized_odds_measure_TP(dataXy, clf, [sensible_feature], ylabel=1)
        print('DEO:', np.abs(EO[sensible_feature][sensible_feature_values[0]] -
                              EO[sensible_feature][sensible_feature_values[1]]))
        acc_list.append(1.0 - accuracy_score(y, pred))
        deo_list.append(np.abs(EO[sensible_feature][sensible_feature_values[0]] -
                              EO[sensible_feature][sensible_feature_values[1]]))

    # Linear FERM
    print('LINEAR FERM')
    for c in C_list:
        print('C:', c)
        list_of_sensible_feature = X[:, sensible_feature]
        clf = svm.SVC(kernel='linear', C=c)
        algorithm = Linear_FERM(dataXy, clf, X[:, sensible_feature])
        algorithm.fit()

        # Accuracy and Fairness
        pred = algorithm.predict(X)
        print('Accuracy fair:', accuracy_score(y, pred))
        # Fairness measure
        EO = equalized_odds_measure_TP(dataXy, algorithm, [sensible_feature], ylabel=1)
        print('DEO fair:', np.abs(EO[sensible_feature][sensible_feature_values[0]] -
                                  EO[sensible_feature][sensible_feature_values[1]]))
        acc_fair_list.append(1.0 - accuracy_score(y, pred))
        deo_fair_list.append(np.abs(EO[sensible_feature][sensible_feature_values[0]] -
                              EO[sensible_feature][sensible_feature_values[1]]))

    plt.figure(2)
    plt.scatter(acc_list, deo_list, marker='o', s=point_size, edgecolors='k', label='Linear SVM', alpha=alpha)
    plt.scatter(acc_fair_list, deo_fair_list, marker='s', s=point_size, edgecolors='k', label='Lienar FERM', alpha=alpha)
    plt.legend()
    plt.xlabel('Misclassification Error')
    plt.ylabel('DEO')
    plt.title("LINEAR SVM vs LINEAR FERM (different C values)")

    plt.show()


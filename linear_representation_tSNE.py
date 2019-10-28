from load_data import load_adult, load_toy_test
from linear_ferm import Linear_FERM
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed!



if __name__ == "__main__":
    which_data = 2  # 1 is adult dataset, 2 is our toy data
    if which_data == 1:
        # Load Adult dataset
        dataset_train, dataset_test = load_adult(smaller=True, scaler=False)
        sensible_feature = 9  # GENDER
        sensible_feature_values = sorted(list(set(dataset_train.data[:, sensible_feature])))
        print('Different values of the sensible feature', sensible_feature, ':', sensible_feature_values)
        ntrain = len(dataset_train.target)
    elif which_data == 2:
        # Load toy test
        dataset_train, dataset_test = load_toy_test()
        sensible_feature = len(dataset_train.data[1, :]) - 1
        sensible_feature_values = sorted(list(set(dataset_train.data[:, sensible_feature])))
        print('Different values of the sensible feature', sensible_feature, ':', sensible_feature_values)
        ntrain = len(dataset_train.target)

    # Linear FERM
    list_of_sensible_feature_test = dataset_test.data[:, sensible_feature]
    # We do not need to fit a model -> model=None
    algorithm = Linear_FERM(dataset=dataset_train, model=None, sensible_feature=sensible_feature)
    # Linear fair representation
    new_dataset_train = algorithm.new_representation(dataset_train.data)
    print('New fair representation done!')

    # An initial PCA dimensionality reduction -- if t-SNE is too slow -- can be a good idea...
    # pca = decomposition.PCA(n_components=5)
    # pca.fit(new_dataset_train)
    # new_dataset_train_PCA = pca.transform(new_dataset_train)
    # pca.fit(dataset_train.data)
    # dataset_train_PCA = pca.transform(dataset_train.data)

    positive = dataset_train.target == 1
    negative = dataset_train.target != 1
    groupA = dataset_train.data[:, sensible_feature] == sensible_feature_values[0]
    groupB = dataset_train.data[:, sensible_feature] != sensible_feature_values[0]
    positive_groupA = positive * groupA
    positive_groupB = positive * groupB
    negative_groupA = negative * groupA
    negative_gorupB = negative * groupB

    X_original = dataset_train.data
    X_fair = new_dataset_train

    n_components = 3  # 2 or 3 dimensional t-SNE
    method = 'exact'  # ‘barnes_hut’

    nfig = 1
    for title, X in [('Original representation', X_original), ('Fair representation', X_fair)]:
        X_embedded = TSNE(n_components=n_components, perplexity=50.0, verbose=10, n_iter=5000, learning_rate=100,
                          method=method).fit_transform(X)
        print('%d-dimensional t-SNE embedding done!' % n_components)
        only_positive_plot = True  # Plot only the positive examples

        fig = plt.figure(nfig)
        if not only_positive_plot:
            if n_components == 2:
                title += ' - 2d t-SNE'
                plt.scatter(X[positive_groupA, 0], X[positive_groupA, 1], c="r", marker='P', label='Positive Group A')
                plt.scatter(X[positive_groupB, 0], X[positive_groupB, 1], c="g", marker='P', label='Positive Group B')
                plt.scatter(X[negative_groupA, 0], X[negative_groupA, 1], c="r", marker='$-$', label='Negative Group A')
                plt.scatter(X[negative_gorupB, 0], X[negative_gorupB, 1], c="g", marker='$-$', label='Negative Group B')
            elif n_components == 3:
                title += ' - 3d t-SNE'
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(X[positive_groupA, 0], X[positive_groupA, 1], X[positive_groupA, 2], c="r", marker='P',
                           label='Positive Group A')
                ax.scatter(X[positive_groupB, 0], X[positive_groupB, 1], X[positive_groupB, 2], c="g", marker='P',
                           label='Positive Group B')
                ax.scatter(X[negative_groupA, 0], X[negative_groupA, 1], X[negative_groupA, 2], c="r", marker='$-$',
                           label='Negative Group A')
                ax.scatter(X[negative_gorupB, 0], X[negative_gorupB, 1], X[negative_gorupB, 2], c="g", marker='$-$',
                           label='Negative Group B')
        else:
            if n_components == 2:
                title += ' - 2d t-SNE - Positive examples'
                plt.scatter(X[positive_groupA, 0], X[positive_groupA, 1], c="r", marker='P', label='Positive Group A')
                plt.scatter(X[positive_groupB, 0], X[positive_groupB, 1], c="g", marker='P', label='Positive Group B')
            elif n_components == 3:
                title += ' - 3d t-SNE - Positive examples'
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(X[positive_groupA, 0], X[positive_groupA, 1], X[positive_groupA, 2], c="r", marker='P',
                           label='Positive Group A')
                ax.scatter(X[positive_groupB, 0], X[positive_groupB, 1], X[positive_groupB, 2], c="g", marker='P',
                           label='Positive Group B')
        nfig += 1
        plt.title(title)
        plt.legend(loc="lower left")
    plt.show()

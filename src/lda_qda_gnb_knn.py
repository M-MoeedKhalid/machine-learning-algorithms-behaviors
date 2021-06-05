import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from classifier_comparison import compareclassifiers
from helper_code import names, Getvalues


def kgraphs(X, y, cv):
    error_rate = []
    k_range = range(1, 21)
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, p=2)
        scores = cross_val_score(knn, X, y, cv=cv, scoring='accuracy')
        k_scores.append(scores.mean())
    # plot to see clearly
    val, idx = max((val, idx) for (idx, val) in enumerate(k_scores))

    plt.close()
    plt.plot(k_range, k_scores, )
    plt.xticks(list(range(1, max(k_range) + 1)), [str(i) for i in range(1, max(k_range) + 1)])
    plt.xlabel('Value of K for KNN for ' + name)
    plt.ylabel('Cross-Validated Accuracy')
    plt.savefig('svm_k-value_comparisons/Value of K for KNN for ' + name + '.png')
    plt.show()
    return idx


classifier_names = ["LDA", "QDA", "GNB", "kNN",
                    ]

colors = ['r', 'g', 'b', 'y', 'c', 'm']
markers = ['8', 's', 'p', 'P', '*', '+']

for idx, name in enumerate(names):
    df = pd.read_csv(name)
    X = []
    y = []
    for index, row in df.iterrows():
        X.append([row['x1'], row['x2']])
        y.append(int(row['label']))
    X = np.array(X)
    y = np.array(y)
    kf = KFold(n_splits=10)
    lda = LinearDiscriminantAnalysis()
    qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    gnb = GaussianNB()
    neigh = KNeighborsClassifier(n_neighbors=15)
    cv = KFold(n_splits=10)
    plt.close()
    plt.scatter(X[:, 0], X[:, 1], marker=markers[idx], c=y,
                s=25, edgecolor='k', label=name, )
    plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='+')
    plt.title(name)
    plt.savefig(name + '.png')
    plt.show()
    classifiers = [
        lda,
        qda,
        gnb,
        neigh
    ]
    kvalue = kgraphs(X, y, cv)
    print("Min k value of " + name + " was found to be " + str(kvalue))
    neigh = KNeighborsClassifier(n_neighbors=int(kvalue))
    for classifier in classifiers:
        scores = cross_val_predict(classifier, X, y, cv=cv)
        conf_mat = confusion_matrix(y, scores)
        accuracy = accuracy_score(y, scores)
        print(str(classifier) + " for " + name)
        # print(conf_mat)
        Sensitivity, Specificity, PPV, NPV = Getvalues(conf_mat)
        print("Acuuracy : " + str(accuracy) + " Sensitivity : " + str(Sensitivity) +
              " Specificity: " + str(Specificity) + " PPV: " + str(PPV) + " NPV: " + str(NPV))

compareclassifiers(classifiers, classifier_names, names)

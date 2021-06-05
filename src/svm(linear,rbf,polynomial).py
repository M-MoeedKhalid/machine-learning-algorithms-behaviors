import numpy as np
from sklearn import svm
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, cross_val_predict
from helper_code import Getvalues, names
from svm_comparison import plot_svm
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
    linear = svm.SVC(
        kernel='linear',
        C=10,
        gamma=0.001
    )
    polynomial = svm.SVC(
        kernel='poly',
        degree=2
    )
    rbf = svm.SVC(
        kernel='rbf',
    )
    classifiers = [
        linear,
        rbf,
        polynomial,
    ]

    for classifier in classifiers:
        scores = cross_val_predict(classifier, X, y, cv=kf)
        conf_mat = confusion_matrix(y, scores)
        accuracy = accuracy_score(y, scores)
        print(str(classifier) + " for " + name)
        # print(conf_mat)
        Sensitivity, Specificity, PPV, NPV = Getvalues(conf_mat)
        print("Acuuracy : " + str(accuracy) + " Sensitivity : " + str(Sensitivity) +
              " Specificity: " + str(Specificity) + " PPV: " + str(PPV) + " NPV: " + str(NPV))

plot_svm(names, classifiers)
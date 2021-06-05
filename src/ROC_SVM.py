import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import  KFold
from helper_code import names

def roc():
    for name in names:
        df = pd.read_csv(name)
        df = df.sample(frac=1)
        X = []
        y = []
        for index, row in df.iterrows():
            X.append([row['x1'], row['x2']])
            y.append(int(row['label']))
        X = np.array(X)
        y = np.array(y)

        cv = KFold(n_splits=10)
        c1 = svm.SVC(
            kernel='rbf',
            probability=True,
        )
        c2 = svm.SVC(
            # C=1.0,
            kernel='rbf',
            probability=True,
            C=10
        )
        c3 = svm.SVC(
            kernel='rbf',
            probability=True,
            gamma=0.001,
            C=100
        )
        c4 = svm.SVC(
            kernel='rbf',
            probability=True,
            gamma=0.001,
            C=1000
        )

        classifiers = [
            c1, c2, c3, c4
        ]
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        i = 1
        for classifier in classifiers:
            for train, test in cv.split(X, y):
                prediction = classifier.fit(X[train], y[train]).predict_proba(X[test])
                fpr, tpr, t = roc_curve(y[test], prediction[:, 1])
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                i = i + 1

            plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
            mean_tpr = np.mean(tprs, axis=0)
            mean_auc = auc(mean_fpr, mean_tpr)
            plt.plot(mean_fpr, mean_tpr, color='blue',
                     label=r'Mean ROC for ' + name + " gamma=" + str(classifier.gamma) + " C=" + str(classifier.C) + " " +
                           ' (AUC = %0.2f )' % (mean_auc), lw=2, alpha=1)

            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC-AUC for ' + name)
            plt.legend(loc="lower right")

        plt.savefig(name + ' ROC-AUC Curve''.jpeg')
        plt.show()

roc()
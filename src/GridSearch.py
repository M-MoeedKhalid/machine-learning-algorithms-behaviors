import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV, KFold
import pandas as pd
from helper_code import names as datasets


def gridsearch():
    for name in datasets:
        df = pd.read_csv(name)
        X = []
        y = []
        for index, row in df.iterrows():
            X.append([row['x1'], row['x2']])
            y.append(int(row['label']))
        X = np.array(X)
        y = np.array(y)
        parameters = [
            {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001, 'scale'], 'kernel': ['linear']},
            {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001, 'scale'], 'kernel': ['rbf']},
            {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001, 'scale'], 'degree': [3], 'kernel': ['poly']},
        ]
        svc = svm.SVC()
        kf = KFold(n_splits=10)
        grid = GridSearchCV(svc, parameters, cv=kf,
                            verbose=3,
                            scoring='accuracy'
                            )
        grid.fit(X, y)
        print("The best for " + name + "parameters are %s with a score of %0.3f"
              % (grid.best_params_, grid.best_score_))


gridsearch()
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, train_test_split, cross_val_predict, cross_val_score, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import feature_selection as BF
from helper_code import Getvalues, names

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
    svml = make_pipeline(StandardScaler(),
                         LinearSVC(random_state=0, tol=1e-5))

    decision_tree = DecisionTreeClassifier()
    neural_network = MLPClassifier(hidden_layer_sizes=(1,), activation='logistic', solver='lbfgs', max_iter=10000)
    random_forest = RandomForestClassifier()
    classifiers = [
        decision_tree,
        neural_network,
        random_forest

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

BF

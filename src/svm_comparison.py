import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_svm(datasets,classifiers):
    for name in datasets:
        df = pd.read_csv(name)
        X = []
        y = []
        for index, row in df.iterrows():
            X.append([row['x1'], row['x2']])
            y.append(int(row['label']))
        X = np.array(X)
        y = np.array(y)
        h = .02  # step size in the mesh
        # create a mesh to plot in
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # title for the plots
        titles = ['SVC_L ' + name,
                  'SVC-R for ' + name,
                  'SVC-P (degree 2) kernel for ' + name,
                  ]
        for i, clf in enumerate(classifiers):
            clf.fit(X,y)
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            plt.subplot(2, 2, i + 1)
            plt.subplots_adjust(wspace=0.4, hspace=0.4)

            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

            # Plot also the training points
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
            # plt.xlabel('Sepal length')
            # plt.ylabel('Sepal width')
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.xticks(())
            plt.yticks(())
            plt.title(titles[i])
        plt.savefig(name + ' SVM''.jpeg')
        plt.show()
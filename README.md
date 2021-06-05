# machine-learning-algorithms-behaviors
An elaborate project done to learn the behaviors of specific machine learning algorithms on different types of datasets.
The algorithms Accuracy, Specificity, Sensitivity and F1-score are calculated and compared with other algorithms

** This code was written to be used as a baseline for any project. **

### Dataset 
The following are the [datasets](https://scikit-learn.org/stable/datasets/sample_generators.html) that are being used in this experiment:

Circles0.3: 
<img src="https://github.com/M-MoeedKhalid/machine-learning-algorithms-behaviors/blob/main/dataset_plots/circles0.3.csv.png" width="320" height="240">

halfkernel: 
<img src="https://github.com/M-MoeedKhalid/machine-learning-algorithms-behaviors/blob/main/dataset_plots/halfkernel.csv.png" width="320" height="240">

moons1: 
<img src="https://github.com/M-MoeedKhalid/machine-learning-algorithms-behaviors/blob/main/dataset_plots/moons1.csv.png" width="320" height="240">

spiral1: 
<img src="https://github.com/M-MoeedKhalid/machine-learning-algorithms-behaviors/blob/main/dataset_plots/spiral1.csv.png" width="320" height="240">

twoguassians33: 
<img src="https://github.com/M-MoeedKhalid/machine-learning-algorithms-behaviors/blob/main/dataset_plots/twogaussians33.csv.png" width="320" height="240">

twoguassians42: 
<img src="https://github.com/M-MoeedKhalid/machine-learning-algorithms-behaviors/blob/main/dataset_plots/twogaussians42.csv.png" width="320" height="240">


### Code:
To run any of the code first run:
`pip install -r requirements.txt` to install the relevant libraries.

The code can be divided into the following parts:

#### Classifiers:
There are three classifier comparisions file in the repository:

* lda_qda_gnb_knn.py uses:
  * Linear Disrciminant Analysis
  * Quadrative Discriminant Analysis
  * Guassian Naive Bayes
  * K-Nearest Neigbour (The value of k is tested from 1 to 21 and the best value is used). The graphs that are plotted for every value of k can be found in /svm_k-   value_comparisons. The function that does this plotting and testing is kgraphs() and can be found inside the file. 
 And compares the algoirthms with the performance metrics stated above.
 This file can be run using `python lda_qda_gnb_knn.py` 
 
 * svm(linear,rbf,polynomial).py uses all these three kernels of SVM and compares them against each other over the performance metrics stated above.
 * dt_nn_rf.py uses 
  * Neural Network
  * Decision Trees
  * Random Forest
  And compares the algorithms with the performance metrics stated above.
  This file can be run using `python svm(linear,rbf,polynomial).py`

### Plots:
* The classifier_comparison.py can be used with any of the files. It uses a list of classifiers, a list of the names of the classifiers, a list of the dataset paths. and it plots all the classifiers against each other for each dataset. This file needs to be called from inside of one of the above mentioned files.
* The svm_comparison.py takes plots the performance of different SVMs for our dataset. This file can be run using `python svm_comparison.py`
* The ROC_SVM.py files plots an ROC-AUC Curve for different SVMs over the dataset. This file can be run using `python ROC_SVM.py

### Grid Search:
GridSearch.py demonstrates an example of using grid search over an algorithm. It uses SVM and then compares several different kernels with several different parameters and then displays the best parameters and the best score at the end.
This file can be run using `python GridSearch.py`




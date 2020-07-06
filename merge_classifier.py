import numpy as np
import pandas as pd
import warnings
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def fit_one_method(method, X, Y, flip=False, random_state=0):
    # models
    if method == 'svm' or method == 'svm_pca':
        model = svm.SVC(C=1e-2, kernel='linear', random_state= random_state)
    elif method == 'random_forest' or method == 'random_forest_pca':
        model = RandomForestClassifier(n_estimators=50,
                                       random_state=random_state)
    elif method == 'logistic_regression' or method == 'logistic_reg_pca':
        model = LogisticRegression(random_state=random_state)
    elif method == 'naive_bayes' or method == 'naive_bayes_pca':
        model = GaussianNB()
    elif method == 'decision_tree' or method == 'decision_tree_pca':
        model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state= random_state)
    elif method == 'logistic_lasso' or method == 'logistic_lasso_pca':
        model = LogisticRegression(penalty='l1', tol=1, solver='liblinear', random_state= random_state)

    # leave-one-out cross validation
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    y_true = []
    y_est = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        if method.endswith('pca'):
            pca = PCA(0.9)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        model.fit(X_train, Y_train)
        y_est_1 = model.predict(X_test)
        y_true.append(Y_test[0])
        if flip == True:
            y_est.append(1 - y_est_1[0])
        else:
            y_est.append(y_est_1[0])

    #calculate the results
    accuracy = accuracy_score(y_true, y_est)
    f1 = f1_score(y_true, y_est)
    precision = precision_score(y_true, y_est)
    recall = recall_score(y_true, y_est)

    #print out the results
    print("Method = %s, Accuracy = %.3f%%, F1 = %.3f%%, Recall = %.3f%%, Precision = %.3f%% " % (
        method, accuracy * 100.0, f1 * 100.0, precision * 100.0, recall * 100.0))
    print(y_est, "\n")

    return np.array(y_est)

warnings.filterwarnings('ignore')
df = pd.read_csv('/Users/jiaxinhe/Desktop/disease_data(weight+features.csv', header=None)

# define features and target
X = df.iloc[:, 1:2556]
Y = df.iloc[:, 2556]

# remove zero column
X = X.loc[:, (X != 0).any(axis=0)]

X = np.array(X).reshape(len(X), -1)
Y = np.array(Y).reshape(len(Y), -1)

# calls to perform different models on X and Y
print(list(df.iloc[:, 2556]), "\n")
y_est1 = fit_one_method('svm', X, Y)
y_est2 = fit_one_method('random_forest', X, Y)
y_est3 = fit_one_method('logistic_regression', X, Y)
y_est4 = fit_one_method('naive_bayes', X, Y)
y_est5 = fit_one_method('decision_tree', X, Y)
y_est6 = fit_one_method('logistic_lasso', X, Y)

pca = PCA(0.90)
X_pca = pca.fit_transform(X)

# calls to perform different models on X and Y after pca
y_est7 = fit_one_method('svm', X_pca, Y)
y_est8 = fit_one_method('random_forest', X_pca, Y)
y_est9 = fit_one_method('logistic_regression', X_pca, Y)
y_est10 = fit_one_method('naive_bayes', X_pca, Y)
y_est11 = fit_one_method('decision_tree', X_pca, Y)
y_est12 = fit_one_method('logistic_lasso', X_pca, Y)

# calls to perform different models on X and Y with separate pca will be applied to each case
y_est13 = fit_one_method('svm_pca', X, Y)
y_est14 = fit_one_method('random_forest_pca', X, Y)
y_est15 = fit_one_method('logistic_reg_pca', X, Y)
y_est16 = fit_one_method('naive_bayes_pca', X, Y)
y_est17 = fit_one_method('decision_tree_pca', X, Y)
y_est18 = fit_one_method('logistic_lasso_pca', X, Y)
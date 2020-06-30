import numpy as np
import pandas as pd
import warnings
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import preprocessing


def fit_one_method(method, X, Y, flip=False, random_state=98):
    # models
    if method == 'svm':
        model = svm.SVC(C=1e-2, kernel='linear', random_state= 98)
    elif method == 'random_forest':
        model = RandomForestClassifier(n_estimators=50,
                                       random_state=random_state)
    elif method == 'logistic_regression':
        model = LogisticRegression(random_state=random_state)
    elif method == 'naive_bayes':
        model = GaussianNB()
    elif method == 'decision_tree':
        model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state= random_state)
    elif method == 'logistic_lasso':
        model = LogisticRegression(penalty='l1', tol=1, solver='liblinear', random_state= random_state)

    loo = LeaveOneOut()
    loo.get_n_splits(X)
    y_true = []
    y_est = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        model.fit(X_train, Y_train)
        y_est_1 = model.predict(X_test)
        y_true.append(Y_test[0])
        if flip == True:
            y_est.append(1 - y_est_1[0])
        else:
            y_est.append(y_est_1[0])

    accuracy = accuracy_score(y_true, y_est)
    f1 = f1_score(y_true, y_est)
    precision = precision_score(y_true, y_est)
    recall = recall_score(y_true, y_est)

    print("Method = %s, Accuracy = %.3f%%, F1 = %.3f%%, Recall = %.3f%%, Precision = %.3f%% " % (
        method, accuracy * 100.0, f1 * 100.0, precision * 100.0, recall * 100.0))
    print(y_est, "\n")

    return np.array(y_est)

warnings.filterwarnings('ignore')
df = pd.read_csv('/Users/jiaxinhe/Desktop/disease_data(weight+features.csv', header=None)

X = df.iloc[:, 1:2556]
Y = df.iloc[:, 2556]

# remove zero column
X = X.loc[:, (X != 0).any(axis=0)]

X = np.array(X).reshape(len(X), -1)
Y = np.array(Y).reshape(len(Y), -1)

#X_scaled = preprocessing.scale(X)
# plt.plot(X[0,:])
# plt.plot(X_scaled[0,:])
# plt.ylabel('some numbers')
# plt.show()

print(list(df.iloc[:, 2556]), "\n")
y_est1 = fit_one_method('svm', X, Y)
y_est2 = fit_one_method('random_forest', X, Y)
y_est3 = fit_one_method('logistic_regression', X, Y)
y_est4 = fit_one_method('naive_bayes', X, Y)
y_est5 = fit_one_method('decision_tree', X, Y)
y_est6 = fit_one_method('logistic_lasso', X, Y)


# y_est_essemble = (((y_est1+y_est2+y_est3+y_est4+y_est5)/5)>0.5).astype(int)
# y_est = list(y_est_essemble)
# y_true = list(df.iloc[:, 61])
# accuracy = accuracy_score(y_true, y_est)
# f1 = f1_score(y_true, y_est)
# precision = precision_score(y_true, y_est)
# recall = recall_score(y_true, y_est)
# method = 'essemble'
# print("Method = %s, Accuracy = %.3f%%, F1 = %.3f%%, Recall = %.3f%%, Precision = %.3f%%" % (method, accuracy * 100.0, f1 * 100.0, precision * 100.0, recall * 100.0))
# print(list(y_est_essemble))


pca = PCA(0.90)
X_pca = pca.fit_transform(X)

y_pca1 = fit_one_method('svm', X_pca, Y)
y_pca2 = fit_one_method('random_forest', X_pca, Y)
y_pca3 = fit_one_method('logistic_regression', X_pca, Y)
y_pca4 = fit_one_method('naive_bayes', X_pca, Y)
y_pca5 = fit_one_method('decision_tree', X_pca, Y)
y_pca6 = fit_one_method('logistic_lasso', X_pca, Y)

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
#
# pca = PCA(.95)
# pca.fit(X_train)
# X_train = pca.transform(X_train)
# X_test = pca.transform(X_test)
# logisticReg = LogisticRegression(solver='lbfgs')
# logisticReg.fit(X_train, y_train)
# y_pred1 = logisticReg.predict(X_test)
# logisticReg_accuracy = metrics.accuracy_score(y_test, y_pred1)
# logisticReg_f1 = metrics.f1_score(y_test, y_pred1)
# logisticReg_precision = metrics.precision_score(y_test, y_pred1)
# logisticReg_recall = metrics.recall_score(y_test, y_pred1)
# test_score = logisticReg.score(X_test, y_test)
# print("accuracy: ", logisticReg_accuracy)
# print("precision: ", logisticReg_precision)
# print("f1: ", logisticReg_f1)
# print("recall: ", logisticReg_recall)

# logisticLasso = LogisticRegression(penalty='l1', solver='liblinear')
# logisticLasso.fit(X_train, y_train)
# y_pred2 = logisticLasso.predict(X_test)
#
# lasso_accuracy = metrics.accuracy_score(y_test, y_pred2)
# lasso_f1 = metrics.f1_score(y_test, y_pred2)
# lasso_precision = metrics.precision_score(y_test, y_pred2)
# lasso_recall = metrics.recall_score(y_test, y_pred2)
# print("accuracy: ", lasso_accuracy)
# print("precision: ", lasso_precision)
# print("f1: ", lasso_f1)
# print("recall: ", lasso_recall)
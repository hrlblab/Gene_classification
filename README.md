# merge_classifier
## Description
This program contains one method “fit_one_method” which sets the model, performs leave-one-out cross validation on features and target passed as parameters and prints out the evaluation results. 
    
We used six models which are **support vector machine, random forest, naïve bayes, logistic regression, decision tree and logistic lasso**. In our program there are three types of calls to “fit_one_method”.  We first removed zero columns from features. The first type calls “fit_one_method” with models’name, features and target.
```
y_est1 = fit_one_method('svm', X, Y)
y_est2 = fit_one_method('random_forest', X, Y)
y_est3 = fit_one_method('logistic_regression', X, Y)
y_est4 = fit_one_method('naive_bayes', X, Y)
y_est5 = fit_one_method('decision_tree', X, Y)
y_est6 = fit_one_method('logistic_lasso', X, Y)
```
Then we performed principal component analysis on features and we chose to use the first 20 components with 90% of the variance explained. The second type calls “fit_one_method” with models’ name, features after pca and target.
```
pca = PCA(0.90)
X_pca = pca.fit_transform(X)

# calls to perform different models on X and Y after pca
y_est7 = fit_one_method('svm', X_pca, Y)
y_est8 = fit_one_method('random_forest', X_pca, Y)
y_est9 = fit_one_method('logistic_regression', X_pca, Y)
y_est10 = fit_one_method('naive_bayes', X_pca, Y)
y_est11 = fit_one_method('decision_tree', X_pca, Y)
y_est12 = fit_one_method('logistic_lasso', X_pca, Y)
```
Lastly, the third type calls with models’ name, features and target but specifying in models’ name that separate pca will be performed on each round of leave-one-out cross validation.

```
y_est13 = fit_one_method('svm_pca', X, Y)
y_est14 = fit_one_method('random_forest_pca', X, Y)
y_est15 = fit_one_method('logistic_reg_pca', X, Y)
y_est16 = fit_one_method('naive_bayes_pca', X, Y)
y_est17 = fit_one_method('decision_tree_pca', X, Y)
y_est18 = fit_one_method('logistic_lasso_pca', X, Y)
```

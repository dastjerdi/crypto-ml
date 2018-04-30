"""
Script to create models
"""
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import sklearn.metrics as metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import auc

import warnings
warnings.filterwarnings('ignore')

def traditional_models(X_train, y_train, X_test, y_test):
    """
    Applies logistic regression
    :param X_train: Training Set Predictors
    :param X_test: Test Set Predictors
    :param y_train: Test Set response
    :param y_test: Test Set response
    :return: Dataframe with ML technique
    """
    #Logistic regression
    cvals = [1e-20, 1e-15, 1e-10, 1e-5, 1e-3, 1e-1, 1, 10, 100, 10000, 100000]
    logregcv = LogisticRegressionCV(Cs=cvals, cv=5)
    logregcv.fit(X_train, y_train)
    yhat = logregcv.predict(X_test)
    logreg_acc = accuracy_score(y_test, yhat)
    fpr_log, tpr_log, thresholds = metrics.roc_curve(y_test, logregcv.predict_proba(X_test)[:, 1])
    logreg_auc = auc(fpr_log, tpr_log)

    # knn
    ks = [2 ** x for x in range(2, 8)]

    cv_scores = []
    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k)

        scores = cross_val_score(knn, X_train, y_train,
                                 cv=5, scoring="accuracy")
        cv_scores.append(scores.mean())

    opt_k = ks[np.argmax(cv_scores)]
    #print('The optimal value for k is %d, with a score of %.3f.'
    #     % (opt_k, cv_scores[np.argmax(cv_scores)]))

    knn = KNeighborsClassifier(n_neighbors=opt_k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)

    knn.fit(X_train, y_train)
    yhat = knn.predict(X_test)
    knn_acc = accuracy_score(y_test, yhat)
    # Calculating auc on testset
    fpr_knn, tpr_knn, thresholds = metrics.roc_curve(y_test, knn.predict_proba(X_test)[:, 1])
    knn_auc = auc(fpr_knn, tpr_knn)

    # LDA
    lda = LinearDiscriminantAnalysis()
    scores = cross_val_score(lda, X_train, y_train, cv=5)

    lda.fit(X_train, y_train)
    yhat = lda.predict(X_test)
    lda_acc = accuracy_score(y_test, yhat)
    # Calculating auc on testset
    fpr_lda, tpr_lda, thresholds = metrics.roc_curve(y_test, lda.predict_proba(X_test)[:, 1])
    lda_auc = auc(fpr_lda, tpr_lda)

    # QDA
    qda = QuadraticDiscriminantAnalysis()
    scores = cross_val_score(qda, X_train, y_train, cv=5)

    qda.fit(X_train, y_train)
    yhat = qda.predict(X_test)
    qda_acc = accuracy_score(y_test, yhat)
    # Calculating auc on testset
    fpr_qda, tpr_qda, thresholds = metrics.roc_curve(y_test, qda.predict_proba(X_test)[:, 1])
    qda_auc = auc(fpr_qda, tpr_qda)

    # Random Forest
    tree_cnts = [2 ** i for i in range(1, 9)]

    # List to hold the results.
    cv_scores = []

    for tree_cnt in tree_cnts:
        # Train the RF model, note that sqrt(p) is the default
        # number of predictors, so it isn't specified here.
        rf = RandomForestClassifier(n_estimators=tree_cnt)
        scores = cross_val_score(rf, X_train, y_train, cv=5)

        cv_scores.append([tree_cnt, scores.mean()])

    cv_scores = np.array(cv_scores)

    opt_tree_cnt = int(cv_scores[np.argmax(np.array(cv_scores)[:, 1])][0])

    rf = RandomForestClassifier(n_estimators=opt_tree_cnt)
    scores = cross_val_score(rf, X_train, y_train, cv=5)

    rf.fit(X_train, y_train)
    yhat = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, yhat)
    # Calculating auc on testset
    fpr_rf, tpr_rf, thresholds = metrics.roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
    rf_auc = auc(fpr_rf, tpr_rf)

    # ADA Boost
    td = [1, 2]
    trees = [2 ** x for x in range(1, 8)]
    param_grid = {"n_estimators": trees,
                  "max_depth": td,
                  "learning_rate": [0.05]
                  }

    p = np.zeros((len(trees) * len(td), 3))
    k = 0
    for i in range(0, len(trees)):
        for j in range(0, len(td)):
            ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=td[j]), n_estimators=trees[i],
                                     learning_rate=.05)
            p[k, 0] = trees[i]
            p[k, 1] = td[j]
            p[k, 2] = np.mean(cross_val_score(ada, X_train, y_train, cv=5))
            k = k + 1
    x = pd.DataFrame(p)
    x.columns = ['ntree', 'depth', 'cv_score']
    p = x.ix[x['cv_score'].argmax()]
    ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=p[1]),
                             n_estimators=int(p[0]), learning_rate = .05)
    ada.fit(X_train, y_train)
    yhat = ada.predict(X_test)
    ada_acc = accuracy_score(y_test, yhat)

    # Calculating auc on testset
    fpr_ada, tpr_ada, thresholds = metrics.roc_curve(y_test, ada.predict_proba(X_test)[:, 1])
    ada_auc = auc(fpr_ada, tpr_ada)

    # Support Vector Classification
    svc = svm.SVC(kernel='rbf', random_state=0, gamma=1, C=1, probability=True)
    # scores = cross_val_score(svc, X_train, y_train, cv=5)
    svc.fit(X_train, y_train)
    yhat = svc.predict_proba(X_test)[:, 1]
    svm_acc = accuracy_score(y_test, yhat > 0.5)

    # Calculating auc on testset
    fpr_svm, tpr_svm, thresholds = metrics.roc_curve(y_test, svc.predict_proba(X_test)[:, 1])
    svm_auc = auc(fpr_svm, tpr_svm)

    x = pd.DataFrame({'Accuracy':[logreg_acc,knn_acc,lda_acc,qda_acc,rf_acc,ada_acc,svm_acc],
                     'AUC':[logreg_auc,knn_auc,lda_auc,qda_auc,rf_auc,ada_auc,svm_auc]},
                     index=['LogReg','KNN','LDA','QDA','RandomForest','ADABoost','SVM'])
    return x






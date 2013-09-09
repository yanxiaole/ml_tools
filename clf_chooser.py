"""
beating the benchmark @StumbleUpon Evergreen Challenge
__author__ : Abhishek Thakur
"""

# -*- coding: utf-8 -*-
import numpy as np
from time import time

from sklearn import cross_validation
#from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import pandas as p


def benchmark(clf, X, y):
    print('_' * 80)
    print("Training: ")
    print(clf)
    K = 8
    t0 = time()
    score = np.mean(cross_validation.cross_val_score(clf, X, y, cv=K,
                                                     scoring='roc_auc'))
    cv_time = time() - t0
    print("cv time: %0.3fs" % cv_time)
    print("cv score:   %0.3f" % score)
    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, cv_time


def main():
    print "loading data.."
    traindata = np.array(p.read_table('train.tsv'))[:, 2]
    alchemy = np.array(p.read_table('train.tsv'))[:, 3]
    traindata = list(traindata + ' ' + alchemy)
    #testdata = list(np.array(p.read_table('test.tsv'))[:, 2])
    y = np.array(p.read_table('train.tsv'))[:, -1]
    y = y.astype(np.int64)

    tfv = TfidfVectorizer(min_df=1,  max_features=None, strip_accents='unicode',
                          analyzer='word', token_pattern=r'\w{1,}', stop_words='english',
                          ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1)

    '''
    X_all = traindata + testdata
    lentrain = len(traindata)

    print "fitting pipeline"
    tfv.fit(X_all)
    print "transforming data"
    X_all = tfv.transform(X_all)
    X = X_all[:lentrain]
    X_test = X_all[lentrain:]
    '''
    print "transforming data"
    X = tfv.fit_transform(traindata)

    #X_test = tfv.transform(testdata)

    results = []
    for clf, name in (
            (Perceptron(n_iter=50), "Perceptron"),
            (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
            (LinearSVC(dual=False, tol=1e-3), "LinearSVC"),
            (SGDClassifier(alpha=.0001, n_iter=50), "SGDClassifier"),
            (KNeighborsClassifier(n_neighbors=10), "kNN")):
        print('=' * 80)
        print(name)
        results.append(benchmark(clf, X, y))

    '''
    print("Naive Bayes")
    for alpha in [0.033, 0.1, 0.33, 1., 3.3, 10]:
        print('=' * 80)
        print("alpha: %.3f" % alpha)
        results.append(benchmark(MultinomialNB(alpha=alpha), X, y))
        results.append(benchmark(BernoulliNB(alpha=alpha), X, y))
    '''

    print('=' * 80)
    print("LogisticRegression")
    clf = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001,
                                C=1, fit_intercept=True, intercept_scaling=1.0,
                                class_weight=None, random_state=None)
    results.append(benchmark(clf, X, y))

    print('=' * 80)
    print("Ridge Classifier")
    clf = RidgeCV(alphas=[0.033, 0.1, 0.33, 1.0, 3.3, 10.0])
    clf.fit(X, y)
    print("alpha: %.3f" % clf.alpha_)
    results.append(benchmark(RidgeClassifier(alpha=clf.alpha_,
                                             tol=1e-4,
                                             solver="lsqr"),
                             X, y))

    '''
    print "training on full data"
    clf.fit(X, y)
    pred = clf.predict_proba(X_test)[:, 1]
    testfile = p.read_csv('test.tsv', sep="\t", na_values=['?'], index_col=1)
    pred_df = p.DataFrame(pred, index=testfile.index, columns=['label'])
    pred_df.to_csv('benchmark.csv')
    print "submission file created.."
    '''


if __name__ == "__main__":
    main()


import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
from sklearn.model_selection import KFold

categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']

newsgroups_train = fetch_20newsgroups(subset='train',  categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',  categories=categories)

num_train = len(newsgroups_train.data)
num_test  = len(newsgroups_test.data)

# max_features is an important parameter. You should adjust it.
vectorizer = TfidfVectorizer(max_features=3000)

X = vectorizer.fit_transform( newsgroups_train.data + newsgroups_test.data )
X_train = X[0:num_train, :]
X_test = X[num_train:num_train+num_test,:]

Y_train = newsgroups_train.target
Y_test = newsgroups_test.target

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X_train, Y_train)

Y_predict = clf.predict(X_test)

print(Y_test)
print(Y_predict)

ncorrect = 0
for dy in  (Y_test - Y_predict):
	if 0 == dy:
		ncorrect += 1

print('text classification accuracy is {}%'.format(round(100.0*ncorrect/len(Y_test)) ) )



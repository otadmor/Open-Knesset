import numpy as np
# print(__doc__)
# 
# import numpy as np
# import matplotlib.pyplot as plt
#
# from sklearn.datasets import make_multilabel_classification
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.svm import SVC
# from sklearn.preprocessing import LabelBinarizer
# from sklearn.decomposition import PCA
# from sklearn.cross_decomposition import CCA


def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, linestyle, label=label)


def plot_subfigure(X, Y, subplot, title, transform):
    if transform == "pca":
        X = PCA(n_components=2).fit_transform(X)
    elif transform == "cca":
        X = CCA(n_components=2).fit(X, Y).transform(X)
    else:
        raise ValueError

    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])

    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])

    classif = OneVsRestClassifier(SVC(kernel='linear'))
    classif.fit(X, Y)

    plt.subplot(2, 2, subplot)
    plt.title(title)

    zero_class = np.where(Y[:, 0])
    one_class = np.where(Y[:, 1])
    plt.scatter(X[:, 0], X[:, 1], s=40, c='gray')
    plt.scatter(X[zero_class, 0], X[zero_class, 1], s=160, edgecolors='b',
               facecolors='none', linewidths=2, label='Class 1')
    plt.scatter(X[one_class, 0], X[one_class, 1], s=80, edgecolors='orange',
               facecolors='none', linewidths=2, label='Class 2')

    plot_hyperplane(classif.estimators_[0], min_x, max_x, 'k--',
                    'Boundary\nfor class 1')
    plot_hyperplane(classif.estimators_[1], min_x, max_x, 'k-.',
                    'Boundary\nfor class 2')
    plt.xticks(())
    plt.yticks(())

    plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
    plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
    if subplot == 2:
        plt.xlabel('First principal component')
        plt.ylabel('Second principal component')
        plt.legend(loc="upper left")




import numpy
import cPickle as pickle
import sys

print >>sys.stderr, 'loading data'; sys.stderr.flush()
keywords, data = pickle.load(open('histograms_with_tags.pkl', 'rb'))
keywords = sorted(list(keywords))

tags = numpy.array([a['tags'] for a in data])
ids = [a['id'] for a in data]
print >>sys.stderr, 'preparing examples'; sys.stderr.flush()
keyword_dict = {}
for i,w in enumerate(keywords):
	keyword_dict[w] = i

learn_data = numpy.zeros((len(data), len(keywords)))
for i,a in enumerate(data):
	histogram = learn_data[i]
	for k,v in a['histogram'].iteritems():
		histogram[keyword_dict[k]] = v

from sklearn import svm
#

print >>sys.stderr, 'vectorizing labels'; sys.stderr.flush()

try:
    from sklearn.preprocessing import MultiLabelBinarizer
    lb = MultiLabelBinarizer()
except ImportError, e:
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()


TRIM_SAMPLES = len(tags) #/ 10
tags = tags[:TRIM_SAMPLES]
learn_data = learn_data[:TRIM_SAMPLES]

lb.fit(tags)
labels = lb.transform(tags)

print "using\t",TRIM_SAMPLES,"samples"
print "\t",len(keywords),"keywords"
print "\t",len(lb.classes_),"tags"


#plt.figure(figsize=(8, 6))
#plot_subfigure(learn_data, labels, 1, "With unlabeled samples + CCA", "cca")
#plot_subfigure(learn_data, labels, 2, "With unlabeled samples + PCA", "pca")
#plt.subplots_adjust(.04, .02, .97, .94, .09, .2)
#plt.show()

from sklearn.multiclass import OneVsRestClassifier

#single_svc = svm.SVC(kernel='polynomial', C=4)

#single_svc = svm.SVC(kernel='linear', cache_size = 2048)

single_svc = svm.LinearSVC()

#single_svc = svm.SVC(kernel='rbf', cache_size = 2048)

classifier = OneVsRestClassifier(single_svc)

trained_classifier = classifier.fit(learn_data, labels)
from sklearn.externals import joblib
import cPickle as pickle
import os
try:
    os.makedirs('classifier_data')
except OSError, e:
    pass
joblib.dump(trained_classifier, 'classifier_data/linear_svc_classifier.jlb') 
pickle.dump(lb, open("classifier_data/label_binarizer.pkl", "wb"))
#print >>sys.stderr, 'learning data'; sys.stderr.flush()
#trained_classifier = classifier.fit(learn_data[:-10], labels[:-10])
#print >>sys.stderr, 'classifying'; sys.stderr.flush()
#predicted_labels = trained_classifier.predict(learn_data[-10:])
#predicted_tags = lb.inverse_transform(predicted_labels)
#
#from pprint import pprint
#for _predicted_tags, _tags, _d in zip(predicted_tags, tags[-10:], data[-10:]):
#	print _d['id'],'--',
#	for _predicted_tag in _predicted_tags:
#		print _predicted_tag[::-1], ",",
#	print "-",
#	for _tag in _tags:
#		print _tag[::-1], ",",
#	print ""

import IPython; IPython.embed()

#from sklearn import cross_validation
#from sklearn import metrics
#scores = cross_validation.cross_val_score(classifier, learn_data, labels, scoring='f1_weighted')
#print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
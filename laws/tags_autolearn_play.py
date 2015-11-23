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
from sklearn import svm
print >>sys.stderr, 'loading data'; sys.stderr.flush()
keywords, data = pickle.load(open('histograms_with_tags.pkl', 'rb'))

tags = numpy.array([a['tags'] for a in data])
ids = [a['id'] for a in data]
print >>sys.stderr, 'preparing examples'; sys.stderr.flush()
keyword_dict = {}
for i,w in enumerate(keywords):
	keyword_dict[w] = i


learn_data = []
for a in data:
	histogram = [0,] * len(keywords)
	for k,v in a['histogram'].iteritems():
		histogram[keyword_dict[k]] = v
	learn_data.append(numpy.array(histogram))
learn_data = numpy.array(learn_data)	

from sklearn.preprocessing import LabelBinarizer
print >>sys.stderr, 'vectorizing labels'; sys.stderr.flush()

tags = tags[:100]
learn_data = learn_data[:100]

lb = LabelBinarizer()
labels = lb.fit_transform(tags)


#plt.figure(figsize=(8, 6))
#plot_subfigure(learn_data, labels, 1, "With unlabeled samples + CCA", "cca")
#plot_subfigure(learn_data, labels, 2, "With unlabeled samples + PCA", "pca")
#plt.subplots_adjust(.04, .02, .97, .94, .09, .2)
#plt.show()


from sklearn.multiclass import OneVsRestClassifier
def _OneVsRestClassifier_multilabel_score(self, X, y):
    if self.multilabel_:
        from sklearn.metrics import accuracy_score
        return np.array(accuracy_score(yy, XX) for yy, XX in zip(y.transpose(), self.predict(X).transpose())).mean()
    else:
        return super(OneVsRestClassifier, self).score(X, y)
OneVsRestClassifier.score = _OneVsRestClassifier_multilabel_score

from sklearn.svm import LinearSVC


#print >>sys.stderr, 'learning data'; sys.stderr.flush()
#classifier = OneVsRestClassifier(svm.SVC(kernel='linear'))
#trained_classifier = classifier.fit(learn_data[:-10], labels[:-10])
#print >>sys.stderr, 'classifying'; sys.stderr.flush()
#predicted_labels = trained_classifier.predict(learn_data[-10:])
#print zip(predicted_labels, labels[-10:])


from sklearn import cross_validation

def _StratifiedKFold_multilabel_iter_test_indices(self):
    n_folds = self.n_folds
    if len(self.y.shape) > 1:
        y1dim = [int(reduce(lambda o,n: o + str(n), yy, '0'),2) for yy in self.y]
    else:
        y1dim = self.y
    idx = np.argsort(y1dim)
    for i in range(n_folds):
        yield idx[i::n_folds]
cross_validation.StratifiedKFold._iter_test_indices = _StratifiedKFold_multilabel_iter_test_indices



from sklearn import metrics
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', C=1))
scores = cross_validation.cross_val_score(classifier, learn_data, labels) #, scoring='f1_weighted')
print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

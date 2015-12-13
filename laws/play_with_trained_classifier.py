import numpy
from sklearn.externals import joblib
import cPickle as pickle

ATTR_AMOUNT = 10

trained_classifier = joblib.load('classifier_data/linear_svc_classifier.jlb') 
lb = pickle.load(open("classifier_data/label_binarizer.pkl", "rb"))
keywords, data = pickle.load(open('histograms_with_tags.pkl', 'rb'))
keywords = numpy.array(sorted(list(keywords)))


#numpy.abs(trained_classifier.coef_).argsort()[:,:10]
important_keywords = keywords[numpy.abs(trained_classifier.coef_).argsort()[:,-ATTR_AMOUNT:]]
pickle.dump(important_keywords, open('meaningful_keyword_for_tag.pkl', 'wb'))

#for estimator in trained_classifier.estimators_:
#    keywords[numpy.abs(estimator.coef_).argsort()[0][:ATTR_AMOUNT]]

import IPython; IPython.embed()

#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy
from sklearn.externals import joblib
import cPickle as pickle
from collections import defaultdict
important_keywords = pickle.load(open('meaningful_keyword_for_tag.pkl', 'rb'))
lb = pickle.load(open("classifier_data/label_binarizer.pkl", "rb"))
keywords, data = pickle.load(open('histograms_with_tags.pkl', 'rb'))
trained_classifier = joblib.load('classifier_data/linear_svc_classifier.jlb') 
keywords = numpy.array(sorted(list(keywords)))
import os
import codecs
try:
    os.makedirs("important_keywords/tags")
except OSError, e:
    pass
try:
    os.makedirs("important_keywords/keywords")
except OSError, e:
    pass

doc_in_class = defaultdict(list)

print "creating tag dicts"
for a in data:
    for tag in a['tags']:
        doc_in_class[tag].append(a)


REMOVE = [u'"', u"'", u'/', u'\\', u',', u'.', u'!', u'@', u'#', u'$', u'%', u'^', u'&', u'*', u'(', u')', u'-', u'=', u'_', u'+', u'\n', u'[', u']', u'{', u'}', u'|', u'<', u'>', u'?', u'~', u":", u';', u'–' , u'”', u'`']
REMOVE.extend([unicode(i) for i in xrange(10)])

print "creating keyword dict"
keyword_dict = {}
for i,w in enumerate(keywords):
    keyword_dict[w] = i


print "splitting docs"
pp_with_tags = pickle.load(open('pp_with_tags.pkl', "rb"))
pp_dict = {}
for pp in pp_with_tags:
    pp_text, pp_tag, pp_id = pp['text'], pp['tags'], pp['id']
    pp_dict[pp_id] = reduce(lambda o, s: o.replace(s, u' '), REMOVE, pp_text).split() # [reduce(lambda o, s: o.replace(s, u''), REMOVE, w) for w in pp_text.split()]

def output_tag(class_file, w, estimator_max, estimator_min, weight, hyperlink = False):
    orig_weight = weight
    pre = post = ''
    if weight > 0 and estimator_max != 0:
        weight /= estimator_max
        if weight > 0.05 and hyperlink:
            pre = u"<a href='../tags/" + w.replace(u'/', u' ') + u".html' title='" + str(orig_weight) + u"'>"
            post = u"</a>"
        print >>class_file, pre + u"<span style='color:#00%02X00;'>" % (int(255 * weight),) + w + u"</span>" + post + "&nbsp;"
    elif weight < 0 and estimator_min != 0:
        weight /= estimator_min
        if weight > 0.05 and hyperlink:
            pre = u"<a href='../tags/" + w.replace(u'/', u' ') + u".html' title='" + str(orig_weight) + u"'>"
            post = u"</a>"
        print >>class_file, pre + u"<span style='color:#%02X0000;'>" % (int(255 * weight),) + w + u"</span>" + post + "&nbsp;"
    else:
        print >>class_file, u"<span style='color:#000000;'>" + w + u"</span>&nbsp;"

def output_word(class_file, w, estimator_max, estimator_min, estimator, keyword_dict, hyperlink = False):
    weight = estimator.coef_[0, keyword_dict[w]]
    pre = post = ''
    if weight > 0 and estimator_max != 0:
        weight /= estimator_max
        if weight > 0.05 and hyperlink:
            pre = u"<a href='../keywords/" + w.replace(u'/', u' ') + u".html' title='" + str(estimator.coef_[0, keyword_dict[w]]) + u"'>"
            post = u"</a>"
        print >>class_file, pre + u"<span style='color:#00%02X00;'>" % (int(255 * weight),) + w + u"</span>" + post + "&nbsp;"
    elif weight < 0 and estimator_min != 0:
        weight /= estimator_min
        if weight > 0.05 and hyperlink:
            pre = u"<a href='../keywords/" + w.replace(u'/', u' ') + u".html' title='" + str(estimator.coef_[0, keyword_dict[w]]) + u"'>"
            post = u"</a>"
        print >>class_file, pre + u"<span style='color:#%02X0000;'>" % (int(255 * weight),) + w + u"</span>" + post + "&nbsp;"
    else:
        print >>class_file, u"<span style='color:#000000;'>" + w + u"</span>&nbsp;"

print "creating files"
for i, (class_name, important_keyword_for_class, estimator) in enumerate(zip(lb.classes_, important_keywords, trained_classifier.estimators_)):
    print "tag", str(i), "/", str(len(lb.classes_))
    class_file = codecs.open("important_keywords/tags/" + class_name.replace("/", ' ') + ".html", "wt", encoding="utf-8")

    print >>class_file, u"<html dir='rtl'><head><title>Keyword Explanation for Class " + class_name + u"</title></head><body><meta http-equiv='Content-Type' content='text/html;charset=UTF-8'>"
    print >>class_file, u"<h1>" + class_name + u"</h1><br/>"
    print >>class_file, u"<h2>Most influencing words</h2>"
    estimator_min, estimator_max = estimator.coef_[0].min(), estimator.coef_[0].max()
    for keyword in important_keyword_for_class:
        output_word(class_file, keyword, estimator_max, estimator_min, estimator, keyword_dict)
        print >>class_file, u"(", str(estimator.coef_[0, keyword_dict[keyword]]), u")<br/>"
    print >>class_file, u"<table border='1'>"
    for doc in doc_in_class[class_name]:

        print >>class_file, u"<tr><td valign='top'>" + unicode(doc['id']) + u"</td><td valign='top'>"

	for w in pp_dict[doc['id']]:
            output_word(class_file, w, estimator_max, estimator_min, estimator, keyword_dict, hyperlink = True)
        print >>class_file, u"</td></tr>"
    print >>class_file, u"</table></body></html>"
    class_file.close()

coef_abs = numpy.abs(trained_classifier.coef_)
sorted_keywords = coef_abs.argsort(axis = 0)
min_weight, max_weight = trained_classifier.coef_.min(axis = 1), trained_classifier.coef_.max(axis = 1)
for keyword_count, i in enumerate(coef_abs.sum(axis = 0).argsort()[::-1]):
    keyword = keywords[i]
    print "keyword", str(keyword_count), "/", str(len(keywords)), "(", str(i), ")"
    keyword_file = codecs.open("important_keywords/keywords/" + keyword.replace("/", ' ') + ".html", "wt", encoding="utf-8")
    print >>keyword_file, u"<html dir='rtl'><head><title>Tag Explanation for Keyword " + keyword + u"</title></head><body><meta http-equiv='Content-Type' content='text/html;charset=UTF-8'>"
    print >>keyword_file, u"<h1>" + keyword + u"</h1><br/>"
    print >>keyword_file, u"<h2>Most influenced tags</h2>"
    weights = trained_classifier.coef_[:,i]
    class_keywords = sorted_keywords[::-1,i]

    for class_name, weight, max_weight_, min_weight_ in zip(lb.classes_[class_keywords], weights[class_keywords], max_weight[class_keywords], min_weight[class_keywords]):
        if weight != 0:
            output_tag(keyword_file, class_name, max_weight_, min_weight_, weight, hyperlink = True)
            print >>keyword_file, u"<br/>"
    print >>keyword_file, u"</body></html>"
    keyword_file.close()

#!/usr/bin/python
# -*- coding: UTF-8 -*-

from BeautifulSoup import BeautifulSoup

import cPickle as pickle

similar_texts = [
    u"הצעת חוק דומה בעיקרה הונחה",
    u"הצעת חוק זהה הונחה",
    u"הצעות חוק דומות בעיקרן הונחו",
]
pp_without_tags = []
pp_with_tags = []


import subprocess
import sys

def parse_lang_tree(text):
#    p = subprocess.Popen(["python", "/home/o/otadmor/Downloads/hebdepparser/parse.py"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    p = subprocess.Popen("python /home/o/otadmor/Downloads/hebdepparser/code/utils/sentences.py | python /home/o/otadmor/Downloads/hebdepparser/parse.py", shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    p.stdin.write(text.encode('utf-8'))

    p.stdin.flush()
    p.stdin.close()
    a = p.stdout.read()
    res = ''
    while a != '':
        res += a
        a = p.stdout.read()
    p.stdout.close()
    p.wait()
    return res
    
REMOVE = [u'"', u"'", u'/', u'\\', u',', u'.', u'!', u'@', u'#', u'$', u'%', u'^', u'&', u'*', u'(', u')', u'-', u'=', u'_', u'+', u'\n', u'[', u']', u'{', u'}', u'|', u'<', u'>', u'?', u'~', u":", u';', u'–' , u'”', u'`']
REMOVE.extend([unicode(i) for i in xrange(10)])
REMOVE.extend([str(i) for i in xrange(10)])


ACCEPTED_POS = ['NN', 'JJ', 'VB']

#    for w in reduce(lambda o, s: o.replace(s, u' '), REMOVE, t).split():

    
import sys
sys.path.append("/home/o/otadmor/Downloads/hebdepparser/")
#sys.path.append("/home/o/otadmor/Downloads/hebdepparser/code/utils/")
from parse import parse_sent
from cStringIO import StringIO
def parse_lang_tree(text):
    text = '\n'.join([l.strip() for l in reduce(lambda o, s: o.replace(s, s + u'\n'), ['.', '!', '?'], text).splitlines()])
    out = StringIO()
    parse_sent(text, out)
    return out.getvalue() 

for i, pp in enumerate(pickle.load(open('pps.pkl', 'rb'))):
	content_html = pp['content_html']
	parsed_html = BeautifulSoup(content_html)
	
	current_tag = parsed_html.find('p', attrs={'class':'explanation-header'})
	if current_tag is None:
		print "no text for pp", pp['id']
		continue
	current_tag = current_tag.findNext()
	text = ''
	while current_tag is not None and '---' not in current_tag.text:
		
		text += current_tag.text
		current_tag = current_tag.findNext()

	tags = pp['tags']
	
	for similar_text in similar_texts:
		similar_loc = text.find(similar_text)
		if similar_loc != -1:
			text = text[:similar_loc]
			
	text = parse_lang_tree(text)
	
	if len(tags) == 0:
	    pp_without_tags.append({'text' : text, 'id' : pp['id']})
	else:
		pp_with_tags.append({'text' : text, 'id' : pp['id'], 'tags' : tags})
		
		
pickle.dump(pp_without_tags, open("pp_without_tags.pkl", "wb"))
pickle.dump(pp_with_tags, open("pp_with_tags.pkl", "wb"))
	

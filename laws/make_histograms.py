import cPickle as pickle
from collections import defaultdict
import re

split_re = re.compile(r'\s+', re.U)

REMOVE = ['"', "'", '/', '\\', ',', '.', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '=', '_', '+', '\n', '[', ']', '{', '}', '|', '<', '>', '?', '~', ":", ';']
REMOVE.extend([str(i) for i in xrange(10)])
def make_histogram(t):
    dd = defaultdict(int)
    for w in t.split():
        dd[reduce(lambda o, s: o.replace(s, ''), REMOVE, w)] += 1
    return dict(dd)
    

pp_with_tags = pickle.load(open('pp_with_tags.pkl', "rb"))

keywords = set()
data = []
for i, pp in enumerate(pp_with_tags):
    pp_text, pp_tag, pp_id = pp['text'], pp['tags'], pp['id']
    try:
        histogram = make_histogram(pp_text)
    except:
        import traceback; traceback.print_exc()
        print "at pp", i
        import pdb; pdb.set_trace()
    data.append({'histogram' : histogram, 'tags' : pp_tag, 'id' : pp_id})
    keywords |= set(histogram.keys()) # this will add the keys
    
    

pickle.dump((keywords, data), open('histograms_with_tags.pkl', 'wb'))

from BeautifulSoup import BeautifulSoup

import cPickle as pickle

pp_without_tags = []
pp_with_tags = []
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
	
	if len(tags) == 0:
	    pp_without_tags.append({'text' : text, 'id' : pp['id']})
	else:
		pp_with_tags.append({'text' : text, 'id' : pp['id'], 'tags' : tags})
		
pickle.dump(pp_without_tags, open("pp_without_tags.pkl", "wb"))
pickle.dump(pp_with_tags, open("pp_with_tags.pkl", "wb"))
	

import os
from scipy import spatial
from heapq import nlargest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from operator import itemgetter
import re
from sklearn.metrics.pairwise import cosine_similarity
from nltk.util import ngrams
import nltk


#load stopwords
def load_stopword ():
	stopwords = []
	with open('stopword') as ins:
		for line in ins:
			stopwords.append(line.replace('\n',''))
	return stopwords

#load data
def load_vocab():
	dictio = {}
	fileName = []
	allFile = []
	# FOR EACH DATASET
	for data in os.listdir("./clean_doc_no_stem2/"):
		allFile.append(data)
		start = open('./clean_doc_no_stem2/'+ data, 'r')
		content = start.readlines()
		sentence = " ".join(content)
		for word in sentence.split():
			dictio[word] = 1
	return allFile, dictio.keys()

def clean (query):
	query = re.sub('[^A-Za-z0-9 .,]+', '', query)
	query = query.lower()
	clean_query = ''	
	for word in filter(None, re.split("[., ]", query)):
		if word not in stopwords:
			clean_query += (word + ' ')
	return clean_query

def init():
	#vocab = load_vocab()
	#stopwords = load_stopword()
	print 'finish building TFIDF - Space'
	
def getContent (kalimat):
	n = 30
	ret = ''
	#print kalimat
	
	kalimat = re.sub('[^A-Za-z0-9 .,]+', '', kalimat)
	kalimat = kalimat.lower()
	for word in kalimat.split():
		ret += word + ' '
		n = n - 1
		if n == 0:
			break
	return ret

def getVectorTFIDF():
	cv_tfidf = TfidfVectorizer(vocabulary=vocab)
	train_set = []
	all_sentence = ''
	#dokumen
	for f_part in allFile:
		start = open('./clean_doc_no_stem2/'+ f_part, 'r')
		content = start.readlines()
		sentence = " ".join(content)
		all_sentence += (' ' + sentence)
		#import ipdb; ipdb.set_trace()
		train_set.append(sentence)
		
	bigram = ngrams(all_sentence.split(), 2)
	return cv_tfidf, train_set, bigram
	
stopwords = load_stopword ()
allFile, vocab = load_vocab()
cv_tfidf, train_set, bigram = getVectorTFIDF()
tfidf_matrix_train = cv_tfidf.fit_transform(train_set)  #finds the tfidf score with normalization

fdist = nltk.FreqDist(bigram)
keyNya = sorted(fdist, key=fdist.get)

import time
def search(words, nRank = 7):
	start = time.time()

	query = clean(words)
	score = {}
	
	cv_tfidf = TfidfVectorizer(vocabulary=vocab)
	#compute frequency distribution for all the bigrams in the text
	
	final_query = ''	
	qu = query.split()
	for i in range (0, len(qu)):
		if i == 0:
			ada = 0
			for key in keyNya:
				if qu[i] == key[1] and fdist[key[1]] > 2:
					final_query += (' ' + " ".join(key))
					#print key
					ada = 1
					break
			if ada == 0:
				final_query += (' ' + qu[i])
		if i == len(qu[i])-1:
			ada = 0
			for key in keyNya:
				if qu[i] == key[0] and fdist[key[0]] > 2:
					final_query += (' ' + " ".join(key))
					#print key
					ada = 1
					break
			if ada == 0:
				final_query += (' ' + qu[i])
		else:
			final_query += (' ' + qu[i])
	query_vector = cv_tfidf.fit_transform([final_query]) 		
	#train_set.insert(0, final_query)
	#import ipdb; ipdb.set_trace()
	
	#tfidf_matrix_train.insert(0, query_vector)
	#import ipdb; ipdb.set_trace()
	#train_set.pop(0)
	res = cosine_similarity(query_vector, tfidf_matrix_train) 
	#tfidf_matrix_train.pop(0)
	#import ipdb; ipdb.set_trace()
	
	print (time.time()-start)
	
	i = 0
	for f_part in allFile:
		f_part = f_part.replace('\n', '')
		score [f_part] = res[0][i]
		i += 1
	
	rel_result = []
	#for name, n in nlargest(50, score.iteritems(), key=itemgetter(1)):
	#	print name, nn
	
	# maximum 50
	print '\nSEARCH RESULT\n'
	rank = 1;
	for name, n in nlargest(50, score.iteritems(), key=itemgetter(1)):
		#open the result:
		fi = open('./doc2/'+ name, 'r')
		print '\n' + str(rank)+". Title: "+ fi.readline().replace('\n', '')
		print 'Weight: ' + str (n)
		fi.readline()
		fi.readline()
		print getContent (" ".join(fi.readlines()))
		fi.close()
		rank += 1
		if rank > nRank:
			print ('\n')
			break
		
	
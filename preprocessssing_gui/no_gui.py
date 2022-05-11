import pandas as pd 
import nltk
from tashaphyne.stemming import ArabicLightStemmer
# path = 'data_set.csv'

def Cleaning(copy_data):
    copy_data['class']=copy_data['class'].replace({'pos':0,'neg':1})
    copy_data.drop(copy_data.columns[copy_data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True) #remove unnamed columns . 
    copy_data['tokens'] = 0
    copy_data['sentenses'] = 0 
    copy_data['sentenses_len'] = 0
    copy_data['without_stopwords'] = 0
    copy_data['light_stem'] = 0
    copy_data['root_stem'] = 0 
    return copy_data

def read_data_set(path):
    data=pd.read_csv(path)
    copy_data = data.copy()
    clean_copy = Cleaning(copy_data)
    return clean_copy
def tokenization(x): # take data copy
    for i in range(100) : 
    	tokens = nltk.word_tokenize(x['text'][i])
    	x['tokens'][i] = tokens
def segmentation(x): # take data copy 
	for i in range(100):
		s_tokens = nltk.data.load('tokenizers/punkt/english.pickle')
		sentens = s_tokens.tokenize(x['text'][i])
		x['sentenses_len'][i] = len(sentens)
		x['sentenses'][i] = sentens
def drop_stop_words(x): #take data copy
	for i in range(100):
		arb_stop_words = set(nltk.corpus.stopwords.words("arabic"))
		tokensOfpureTextWithoutstop=[token for token in x['tokens'][i] if token not in arb_stop_words]
		x['without_stopwords'][i] = tokensOfpureTextWithoutstop     
#tashfeen light streamer
def stemming_Light(x):
    ArListem = ArabicLightStemmer()
    for i in range(100):
    		stemming_Light =[ArListem.light_stem(token) for token in x['without_stopwords'][i] ]
    		x['light_stem'][i] = stemming_Light

# ISRIStemmer root-based
def stemming(x):
	st = nltk.ISRIStemmer()
	for i in range(100):
    		stemming_root =[st.stem(token) for token in x['without_stopwords'][i] ]
    		x['root_stem'][i] = stemming_root
	


#-----"un-comment-this-to-run"-----# 
# copy = read_data_set(path)
# data_copy = Cleaning(copy)
# tokenization(data_copy) 
# segmentation(data_copy)
# drop_stop_words(data_copy)
# stemming_Light(data_copy)
# stemming(data_copy)
# print(data_copy['tokens'][10])
# print(data_copy['sentenses'][10])
# print(data_copy['without_stopwords'][10])
# print(data_copy['light_stem'][10])
# print(data_copy['root_stem'][10])
# print(data_copy.head())

# data_copy.to_csv('newdata.csv') #save our work 


#--------------------------#
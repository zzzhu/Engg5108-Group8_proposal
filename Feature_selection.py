
# coding: utf-8

import pandas as pd
import Data_preprocessing
import nltk
import nltk.corpus
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf2 = TfidfVectorizer()
tfidf_ngram=TfidfVectorizer(stop_words='english',ngram_range=(1,4),use_idf=True,smooth_idf=True)

#TF-IDF Frequency Feature Extraction Model
#TF-IDF method I
def Tfidf1(data):
    data_tfidf=transformer.fit_transform(vectorizer.fit_transform(data['Statement'].values))
    return data_tfidf
data_tfidf1=Tfidf1(Data_preprocessing.train_news)
#print(data_tfidf1)

#TF-IDF method II
def Tfidf2(data):
    data_tfidf=tfidf2.fit_transform(data['Statement'].values)# tfidf2 can be replaced with tfidf_ngram to achieve bag of words based on n_gram
    return data_tfidf
data_tfidf2=Tfidf2(Data_preprocessing.train_news)
#print(data_tfidf2)

#TF-IDF method III
def Tfidf3(data):
    data_tfidf=tfidf_ngram.fit_transform(data['Statement'].values)
    return data_tfidf
data_tfidf3=Tfidf3(Data_preprocessing.train_news)
#print(data_tfidf3)

'''
def Vectorize(data):
    data_count=vectorizer.fit_transform(data['Statement'].values)
    #print(data_count)
    #print(type(data_count)) # <class 'scipy.sparse.csr.csr_matrix'> 
    #data_count.shape # get size
    #print(vectorizer.vocabulary_) # check vocabulart
    #print(vectorizer.get_feature_names()[:25])# get features' name
    return data_count
data_count=vectorize(train_news)
'''

'''
def Tfidf_transform(data_count):
    data_tfidf=transformer.fit_transform(data_count)
    #print(data_tfidf)
    #print(type(data_tfidf)) 
    #data_tfidf.shape # get size
    return data_tfidf
data_count=Tfidf_transform(data_count)
'''


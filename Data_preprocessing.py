
# coding: utf-8

# In[68]:


import pandas as pd
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from nltk.util import ngrams
english_stemmer=SnowballStemmer('english')
stopwords=set(nltk.corpus.stopwords.words('english'))

#Read data files
train_data='train.csv'
test_data='test.csv'
valid_data='valid.csv'

train_news=pd.read_csv(train_data)
test_news=pd.read_csv(test_data)
valid_news=pd.read_csv(valid_data)

def observ_data():
    print('---Training dataset size:')
    print("--",train_news.shape)
    print("--",train_news.head(5))
    
    #Datasets below are used to test model and validate result
    print('\n---Testing dataset size:')
    print("--",test_news.shape)
    print("--",test_news.head(5))
    
    print('\n---Validation dataset size:')
    print("--",valid_news.shape)
    print("--",valid_news.head(5))
#Excute the function by calling command below
#observ_data()

#Look into Datasets
def plot_distribution(plot_data,name):
    type_file=plot_data.groupby('Label')['Label'].count()
    sent_type=type_file.index.tolist()
    sent_freq=type_file.values.tolist()
    sent_type1=[str(x) for x in sent_type]
    plt.bar(sent_type1,sent_freq)
    plt.title('Frequency of types')
    plt.xlabel('Type')
    plt.ylabel('Frequency')
    plt.savefig('./Distribution of %s.png'%name)
    plt.close()
#Plot distribution of different datasets by excuting commands below
#plot_distribution(train_news,'train')
#plot_distribution(test_news,'test')
#plot_distribution(valid_news,'validation')

#Stemming
#Word Tokenizer
def tokenize_state(dataset):
    tokens=[[word for word in word_tokenize(state)] for state in dataset['Statement']]
    tokens=[[english_stemmer.stem(word) for word in token] for token in tokens]
    tokens=[[word for word in token if word not in stopwords] for token in tokens]#Remove stopwords
    return tokens
#tokens=tokenize_state(test_news)
#tokens?

#Creating n-grams
#unigram
def create_unigram(words):
    assert type(words)== list
    return words

def create_bigram(words):
    assert type(words)==list
    skip=0
    join_str=" "
    Len=len(words)
    if Len>1:
        lst=[]
        for i in range(Len-1):
            for k in range(1,skip+2):
                if i+k<Len:
                    lst.append(join_str.join([words[i],words[i+k]]))
    else:
        lst=create_unigram(words)
    return lst
#lst=create_bigram(tokens[0])
#lst

#nltk.util.ngrams can also be used to generate ngrams
#list1=ngrams(tokens[0],3)
#for i in list1:
#    print(i)


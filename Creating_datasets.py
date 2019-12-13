
# coding: utf-8

# Creating Datasets
import pandas as pd
import numpy as np
from numpy import nan as NA
from sklearn.model_selection import train_test_split

#Checking Dataset integerity
def dataqual_check(dataset):
    print('---Checking intergrity---')
    dataset.isnull().sum()
    dataset.info()
    print('---Checking Completed')
#dataqual_check(dataset)

#Loading dataset
def load_data():
    raw_data=pd.read_csv('news_sample.csv')
    pdata={'Type':raw_data['type'],
           'Statement':raw_data['title']}
    file=pd.DataFrame(pdata)
    cols=['Statement','Type']
    file=file.loc[:,cols]
    dataqual_check(file)
    print('---Raw data is loaded')
    return file
#Calling command below to load data and names as loaded_data
#loaded_data=load_data()
#loaded_data

#Cleaned data
def clean_data(loaded_data):
    cleaned_data=loaded_data.dropna()
    return cleaned_data
#cleaned_data=clean_data(loaded_data)
#cleaned_data

#Building training dataset,testing dataset and predicting dataset
def build_datasets(cleaned_data):
    experi_data1=cleaned_data[(cleaned_data.Type=='fake')|(cleaned_data.Type=='reliable')|(cleaned_data.Type=='conspiracy')]
    pdata1={'Type':experi_data1['Type']=='reliable',
           'Statement':experi_data1['Statement']}
    experi_data=pd.DataFrame(pdata1)
    #experi_data.to_csv('experi_data.csv',na_rep='NULL',encoding='utf-8')
    #test_read=pd.read_csv('experi_data.csv')
    #test_read
    train_data,test_data = train_test_split(experi_data,test_size = 0.2)
    return train_data,test_data
#train_data,test_data=build_datasets(cleaned_data)
#train_data


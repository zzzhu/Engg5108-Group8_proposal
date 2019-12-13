
# coding: utf-8
import Data_preprocessing
import Feature_selection
import numpy as np
import pickle
import matplotlib as plt
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

#Using Bag of words features
#Building Naive Bayes Classifier
NB_pipeline=Pipeline([('NBvectorizer',Feature_selection.vectorizer),
                      ('NBclassifier',MultinomialNB())])
#NB_pipeline.fit(Data_preprocessing.train_news['Statement'],Data_preprocessing.train_news['Label'])
#predicted_NB=NB_pipeline.predict(Data_preprocessing.test_news['Statement'])
#np.mean(predicted_NB==Data_preprocessing.test_news['Label'])

#Building Logistic Regression Classifier
LR_pipeline=Pipeline([('LRvectorizer',Feature_selection.vectorizer),
                      ('LRclassifier',LogisticRegression())])
#LR_pipeline.fit(Data_preprocessing.train_news['Statement'],Data_preprocessing.train_news['Label'])
#predicted_LR=LR_pipeline.predict(Data_preprocessing.test_news['Statement'])
#np.mean(predicted_LR==Data_preprocessing.test_news['Label'])


#Building Linear SVM Classifier
SVM_pipeline=Pipeline([('SVMvectorizer',Feature_selection.vectorizer),
                       ('SVMclassifier',svm.LinearSVC())])
#SVM_pipeline.fit(Data_preprocessing.train_news['Statement'],Data_preprocessing.train_news['Label'])
#predicted_SVM=SVM_pipeline.predict(Data_preprocessing.test_news['Statement'])
#np.mean(predicted_SVM==Data_preprocessing.test_news['Label'])


#Building Linear SGD Classifier
SGD_pipeline=Pipeline([('SGDvectorizer',Feature_selection.vectorizer),
                       ('SGDclassifier',SGDClassifier(loss='hinge',alpha=1e-3,max_iter=5))])
#SGD_pipeline.fit(Data_preprocessing.train_news['Statement'],Data_preprocessing.train_news['Label'])
#predicted_SGD=SGD_pipeline.predict(Data_preprocessing.test_news['Statement'])
#np.mean(predicted_SGD==Data_preprocessing.test_news['Label'])

#Building Random Foreset Classifier
RF_pipeline=Pipeline([('RFvectorizer',Feature_selection.vectorizer),
                      ('RFclassifier',RandomForestClassifier(n_estimators=200,n_jobs=3))])
#RF_pipeline.fit(Data_preprocessing.train_news['Statement'],Data_preprocessing.train_news['Label'])
#predicted_RF=RF_pipeline.predict(Data_preprocessing.test_news['Statement'])
#np.mean(predicted_RF==Data_preprocessing.test_news['Label'])

#Improve performance by using K-Fold Cross Validation
#Build Confusion Matrix to analysis the performance of different Classifiers
def build_confusion_matrix(classifier,split_num):
    k_fold=KFold(n_splits=split_num)
    scores=[]
    Confusion_Matrix=[[0,0],[0,0]]
    for train_ind,test_ind in k_fold.split(Data_preprocessing.train_news):
        train_text=Data_preprocessing.train_news.iloc[train_ind]['Statement']
        train_y=Data_preprocessing.train_news.iloc[train_ind]['Label']
        test_text=Data_preprocessing.train_news.iloc[test_ind]['Statement']
        test_y=Data_preprocessing.train_news.iloc[test_ind]['Label']
        classifier.fit(train_text,train_y)
        predictions=classifier.predict(test_text)
        Confusion_Matrix+=confusion_matrix(test_y,predictions)
        scores.append(f1_score(test_y,predictions))
    return (print('Total statements classified:', len(Data_preprocessing.train_news)),
            print('Score:', sum(scores)/len(scores)),
            print('score length', len(scores)),
            print('Confusion matrix:'),
            print(Confusion_Matrix))
#build_confusion_matrix(classifier,split_num)
#build_confusion_matrix(NB_pipeline,7)
#build_confusion_matrix(LR_pipeline,6)
#build_confusion_matrix(SVM_pipeline,7)
#build_confusion_matrix(RF_pipeline,6)

#Using N-grams
#Building Naive Bayes Classifier
NB_pipeline_ngram=Pipeline([('NBnvectorizer',Feature_selection.tfidf_ngram),
                      ('NBnclassifier',MultinomialNB())])
#NB_pipeline_ngram.fit(Data_preprocessing.train_news['Statement'],Data_preprocessing.train_news['Label'])
#predicted_NB_ngram=NB_pipeline_ngram.predict(Data_preprocessing.test_news['Statement'])
#np.mean(predicted_NB_ngram==Data_preprocessing.test_news['Label'])

#Building Logistic Regression Classifier
LR_pipeline_ngram=Pipeline([('LRnvectorizer',Feature_selection.tfidf_ngram),
                      ('LRnclassifier',LogisticRegression())])
#LR_pipeline_ngram.fit(Data_preprocessing.train_news['Statement'],Data_preprocessing.train_news['Label'])
#predicted_LR_ngram=LR_pipeline_ngram.predict(Data_preprocessing.test_news['Statement'])
#np.mean(predicted_LR_ngram==Data_preprocessing.test_news['Label'])

#Building Linear SVM Classifier
SVM_pipeline_ngram=Pipeline([('SVMvectorizer',Feature_selection.tfidf_ngram),
                       ('SVMclassifier',svm.LinearSVC())])
#SVM_pipeline_ngram.fit(Data_preprocessing.train_news['Statement'],Data_preprocessing.train_news['Label'])
#predicted_SVM_ngram=SVM_pipeline_ngram.predict(Data_preprocessing.test_news['Statement'])
#np.mean(predicted_SVM_ngram==Data_preprocessing.test_news['Label'])

#Building Linear SGD Classifier
SGD_pipeline_ngram=Pipeline([('SGDvectorizer',Feature_selection.tfidf_ngram),
                       ('SGDclassifier',SGDClassifier(loss='hinge',alpha=1e-3,max_iter=5))])
#SGD_pipeline_ngram.fit(Data_preprocessing.train_news['Statement'],Data_preprocessing.train_news['Label'])
#predicted_SGD_ngram=SGD_pipeline_ngram.predict(Data_preprocessing.test_news['Statement'])
#np.mean(predicted_SGD_ngram==Data_preprocessing.test_news['Label'])

#Building Random Foreset Classifier
RF_pipeline_ngram=Pipeline([('RFvectorizer',Feature_selection.tfidf_ngram),
                      ('RFclassifier',RandomForestClassifier(n_estimators=200,n_jobs=3))])
#RF_pipeline_ngram.fit(Data_preprocessing.train_news['Statement'],Data_preprocessing.train_news['Label'])
#predicted_RF_ngram=RF_pipeline_ngram.predict(Data_preprocessing.test_news['Statement'])
#np.mean(predicted_RF_ngram==Data_preprocessing.test_news['Label'])

#build_confusion_matrix(NB_pipeline_ngram,5)
#build_confusion_matrix(LR_pipeline_ngram,6)
#build_confusion_matrix(SVM_pipeline_ngram,6)
#build_confusion_matrix(SGD_pipeline_ngram,6)
#build_confusion_matrix(RF_pipeline_ngram,6)

#print(classification_report(Data_preprocessing.test_news['Label'], predicted_NB_ngram))
#print(classification_report(Data_preprocessing.test_news['Label'], predicted_LR_ngram))
#print(classification_report(Data_preprocessing.test_news['Label'], predicted_SVM_ngram))
#print(classification_report(Data_preprocessing.test_news['Label'], predicted_SGD_ngram))
#print(classification_report(Data_preprocessing.test_news['Label'], predicted_RF_ngram))

#Choose LR and SVM as candidate models
#Find optimal parameters for LR
'''
LR_parameters = {'LR_tfidf__ngram_range': [(1,1),(1, 2),(1,3),(1,4),(1,5)],
                 'LR_tfidf__use_idf': (True, False),
                 'LR_tfidf__smooth_idf': (True, False)}
LR_gop = GridSearchCV(LR_pipeline_ngram, parameters, n_jobs=-1)
LR_gop = LR.gop.fit(Data_preprocessing.train_news['Statement'][:10000],Data_preprocessing.train_news['Label'][:10000])
#LR_gop.best_score_
#LR_gop.best_params_
#LR_gop.cv_results_
'''

#Find optimal parameters for SVM
'''
SVM_parameters = {'SVM_tfidf__ngram_range': [(1,1),(1,2),(1,3),(1,4),(1,5)],
                  'SVM_tfidf__use_idf': (True, False),
                  'SVM_tfidf__smooth_idf': (True, False),
                  'SVM_clf__penalty': ('l1','l2'),}
SVM_gop= GridSearchCV(SVM_pipeline_ngram, parameters, n_jobs=-1)
SVM_gop= SVM_gop.fit(DataPrep.train_news['Statement'][:10000],DataPrep.train_news['Label'][:10000])
#LR_gop.best_score_
#LR_gop.best_params_
#LR_gop.cv_results_
'''

LR_pipeline_final = Pipeline([
    ('LR_tfidf',TfidfVectorizer(stop_words='english',ngram_range=(1,5),use_idf=True,smooth_idf=False)),
    ('LRclassifier',LogisticRegression(penalty="l2",C=1))
])


#Testing Final Model
LR_pipeline_final.fit(Data_preprocessing.train_news['Statement'],Data_preprocessing.train_news['Label'])
#predicted_LR_final = LR_pipeline_final.predict(Data_preprocessing.test_news['Statement'])
#np.mean(predicted_LR_final == Data_preprocessing.test_news['Label'])
#0.6177

#print(classification_report(Data_preprocessing.test_news['Label'], predicted_LR_final))

#Saving chosen model
#file = 'final_model.sav'
#pickle.dump(LR_pipeline_ngram,open(file,'wb'))


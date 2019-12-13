
# coding: utf-8

import Classifier
import Data_preprocessing
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve

#Plot learning curve
def plot_learning_curve(pipeline,title,fold_size=10000):
    cv=KFold(fold_size, shuffle=True)
    X=Data_preprocessing.train_news['Statement']
    y=Data_preprocessing.train_news['Label']
    pipeline.fit(X,y)
    train_sizes, train_scores, test_scores = learning_curve(pipeline,X,y,n_jobs=-1,cv=cv,train_sizes=np.linspace(.1,1.0,5), verbose=0)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.figure()
    plt.title(title)
    plt.legend(loc="best")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.gca().invert_yaxis()
    #plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.ylim(-.1,1.1)
    plt.show()
#plot_learning_curve(Classifier.LR_pipeline_ngram,"LogisticRegression Classifier")

#plot Precision-Recall curve
def plot_PRcurve(classifier):
    precision, recall, thresholds = precision_recall_curve(Data_preprocessing.test_news['Label'], classifier)
    
    average_precision = average_precision_score(Data_preprocessing.test_news['Label'], classifier)
    plt.step(recall, precision, color='b', alpha=0.2,where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Logistic Regression'.format(average_precision))
#plot_PRcurve(Classifier.predicted_LR_ngram)


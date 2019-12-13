
# coding: utf-8
import pickle
#Verfication
def fake_news_detection():
    news = input("Please input the news you want to verify: ")
    print("Input: " + str(news))
    load_model = pickle.load(open('best_model.sav','rb'))#load the best model
    prediction = load_model.predict([news])
    prob = load_model.predict_proba([news])
    return (print("The given statement is ",prediction[0]),
        print("The truth probability score is ",prob[0][1]))
fake_news_detection()


import tweepy
import pickle
import numpy as np
import pandas as pd
import csv
import re
import plotly.express as px
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
from flask import Flask, request, redirect, render_template,jsonify


class TweetLoader:
    def __init__(self, hashtag, start_time, consumer_key = '5oO181JGmNWqln0n5rozTxzy7',
                 consumer_secret = '1GvpNMRURMRBJ5sOsHoB0MdxqmD365HKRBKyyptvvIWbP2ZlqX',
                 access_token = '1277638295737032710-WVsABPkBUofg0z3yH6g0iTWBBx0ywZ',
                 access_token_secret = 'hH6lW7UzzX0OyU6QMEfhOWM6E2brqEU35YGzgcS0BXmmE', 
                 ):
        self.hashtag = hashtag
        self.start_time = start_time
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret
        
    def api(self):
        auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
        auth.set_access_token(self.access_token, self.access_token_secret)
        api = tweepy.API(auth)
        return api
    
    def get_data(self):
        api = self.api()
        t = []
        for tweet in tweepy.Cursor(api.search, q = self.hashtag, count=100, lang="en", since=self.start_time).items():
            t.append(tweet.text)
        tweets = pd.DataFrame(t, columns=['tweets']) 
        return tweets['tweets']


class Clean:
    def __init__(self, tweets):
        self.tweets = tweets
        pat1 = r'@[A-Za-z0-9]+'
        pat2 = r'https?://[A-Za-z0-9./]+'
        pat3 = r'RT[\s]+'
        pat4 = r'\r\n'
        pat5 = r'_'
        self.combined_pat = r'|'.join((pat1, pat2, pat3, pat4, pat5))
        
    def cl(self, tweets):
        tweets = BeautifulSoup(tweets, 'lxml').get_text()
        tweets = re.sub(self.combined_pat, ' ', tweets)
        tweets = re.sub('[\W]+', ' ', tweets.lower())
        return tweets
        
    def clean_data(self):
        self.tweets = self.tweets.apply(self.cl)


    def tp(self, tweet):
        porter = PorterStemmer()
        tweet = " ".join([porter.stem(word) for word in tweet.split()])
        return tweet

    def tokenizer_porter(self):
        self.tweets = self.tweets.apply(self.tp)
        
    def delet_duplicates(self):
        self.tweets = self.tweets.drop_duplicates() 
        


class PredictionVisualization:
    def __init__(self, tfidf, count, model):
        self.tfidf = tfidf
        self.count = count
        self.model = model 
        
    def predict(self, data):
        vectorized_data = self.tfidf.transform(self.count.transform(data))
        model_predict = self.model.predict(vectorized_data)
        print(model_predict.shape)
        return model_predict
    
    def histogram(self, data):
        model_predict = self.predict(data)
        fig = px.histogram(model_predict)
        fig.show()
        # plt.hist(model_predict)
        # plt.savefig('static/plot.png')


def hashtag(hashtag):    
    # tweetloader = TweetLoader(hashtag, "2005-01-01")
    # data = tweetloader.get_data()
    data = pd.read_csv("train.csv", header=None, encoding='latin-1')[5]
    data = data.sample(100)       
    cleaner = Clean(data)
    cleaner.clean_data()
    cleaner.tokenizer_porter()
    cleaner.delet_duplicates()
    count = pickle.load(open( "count", "rb" ))
    tdidf = pickle.load(open( "tdidf", "rb" ))
    logistic = pickle.load(open( "logistic", "rb" ))
    predict_model = PredictionVisualization(tdidf,count,logistic)
    predict_model.predict(data)
    predict_model.histogram(data)


app = Flask(__name__, static_folder="./static")

@app.route('/')
def index():
	return render_template("home.html")

# @app.route("/plot")
# def plot():
# 	return render_template("plot.html")

@app.route('/hashtag', methods=['GET','POST'])
def my_form_post():
    text1 = request.form['text1']
    hashtag(text1)
    

if __name__ == '__main__':
	app.run(debug=True)

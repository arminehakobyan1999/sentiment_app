import tweepy
import pickle
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import plotly.express as px
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
from flask import Flask, request, redirect, render_template,jsonify


class Api:
    """"This class create API."""
    def __init__(self, consumer_key = '5oO181JGmNWqln0n5rozTxzy7',
                 consumer_secret = '1GvpNMRURMRBJ5sOsHoB0MdxqmD365HKRBKyyptvvIWbP2ZlqX',
                 access_token = '1277638295737032710-WVsABPkBUofg0z3yH6g0iTWBBx0ywZ',
                 access_token_secret = 'hH6lW7UzzX0OyU6QMEfhOWM6E2brqEU35YGzgcS0BXmmE', 
                 ):
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret
        
    @property
    def api(self):
        auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
        auth.set_access_token(self.access_token, self.access_token_secret)
        api = tweepy.API(auth, wait_on_rate_limit=True)
        return api
        
class TweetLoader:
    """TweetLoder download tweets
    Parameters
    ------------
    api : 'tweepy.api.API'
    hashtag: str
    """
    def __init__(self, api, hashtag ):
        self.hashtag = hashtag
        self.api = api

    def get_data(self):
        t = []
        for tweet in tweepy.Cursor(self.api.search, q = self.hashtag, count=100, lang="en", since="2020-01-01").items():
            t.append(tweet.text)
        tweets = pd.DataFrame(t, columns=['tweets'])
        return tweets['tweets']
         

class Clean:
    """"Clean class is clean data
    Parameters
    ------------
    tweets: Data Frame 
    """

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
    """PredictionVisualization 
    Parameters
    ------------
    model: pickle file 
    data: Data Frame
    """
    def __init__(self,  model):
        self.model = model 
        
    def predict(self, data):
        model_predict = np.argmax(self.model.predict_proba(data), axis=1)
        model_predict = [i*'Positive' + (1-i)*'Negative' for i in model_predict]
        return model_predict
    
    def histogram(self, data):
        model_predict = self.predict(data)
        print(model_predict)
        fig = px.histogram(model_predict)
        fig.show()

def hashtag(hashtag, api, model):
    tweetloader = TweetLoader(api ,hashtag)
    data = tweetloader.get_data()       
    cleaner = Clean(data)
    cleaner.clean_data()
    cleaner.tokenizer_porter()
    cleaner.delet_duplicates()
    predict_model = PredictionVisualization(model)
    predict_model.histogram(data)

    
app = Flask(__name__)

@app.before_first_request
def call_api():
    api = Api()
    app.api = api.api
    app.model = pickle.load(open("modell", "rb"))
    print(app.model.__dir__())

@app.route('/')
def index():
    return render_template("home.html")


@app.route('/hashtag', methods=['GET','POST'])
def my_form_post():
    text1 = request.form['text1']
    hashtag(text1, app.api, app.model)
    

if __name__ == '__main__':
    app.run(debug=True)

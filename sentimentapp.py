import datetime as dt
import re
import csv
import requests

import numpy as np
import pandas as pd
import streamlit as st
from pytorch_pretrained_bert.modeling import *
from pytorch_pretrained_bert.optimization import *
from bert_sentiment_utils import predict
from textblob import TextBlob
from newspaper import Article
import matplotlib.pyplot as plt

#@st.cache
def load_model():
    model = BertForSequenceClassification.from_pretrained('finbert-sentiment', cache_dir=None,  num_labels=3)
    return model

def main():
    ## SET OPTIONS
    pd.set_option('display.max_colwidth', 100)

    # Set page title
    st.title('Financial News Sentiment Analyzer')

    # Load classification model
    with st.spinner('Loading BERT Sentiment model...'):
        model = load_model()

    st.subheader('Single Sentence classification')
    tweet_input = st.text_input('Tweet:')

    if tweet_input != '':

        # Make predictions
        with st.spinner('Predicting...'):
            prediction = predict(tweet_input, model)
            blob = TextBlob(tweet_input)
            prediction['textblob_prediction'] = [sentence.sentiment.polarity for sentence in blob.sentences]
        st.write('Prediction:')
        st.dataframe(prediction) 

    st.subheader('Insert Financial News Article URL')

    # Get user input
    url = st.text_input('Enter News Article URL:', '')

    # As long as the query is valid (not empty)
    if url != '':
        try:
            request = requests.get(url)
            with st.spinner(f'Searching for and analyzing {url}...'):
                #Get the article body
                article = Article(url)
                article.download()
                article.parse()
                body = article.text
                #Make prediction using BERT
                prediction = predict(body, model)
                #Make prediction using Textblob
                blob = TextBlob(body)
                prediction['textblob_prediction'] = [sentence.sentiment.polarity for sentence in blob.sentences]
            
            st.write('Prediction:')
            st.dataframe(prediction.style.\
                                    highlight_max(axis=0)\
                                    .background_gradient(cmap='RdYlGn'))
            try:
                #most positive sentence
                st.subheader('Most positive sentence is: \n')
                best = prediction[prediction.sentiment_score == prediction.sentiment_score.max()].sentence.values[0]
                st.write('***"'+best+'"***')
                st.write(f'Score = {prediction.sentiment_score.max()}')
                #Most negative sentence
                st.subheader('Most negative sentence is: \n')
                worst = prediction[prediction.sentiment_score == prediction.sentiment_score.min()].sentence.values[0]
                st.markdown('***"'+worst+'"***')
                st.write(f'Score = {prediction.sentiment_score.min()}')
            except:
                pass
        except Exception as e:
            st.write(e)
            st.write('Web site does not exist. Please try another URL.') 


    st.subheader('PROJECT BY NAZIH KALO') 

if __name__ == '__main__':
    main()
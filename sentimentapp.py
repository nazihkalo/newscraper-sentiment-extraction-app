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
from matplotlib.colors import LinearSegmentedColormap

colors = ["red", "white", "green"]
cmap = LinearSegmentedColormap.from_list("mycmap", colors)
#@st.cache
def load_model():
    model = BertForSequenceClassification.from_pretrained('finbert-sentiment', cache_dir=None,  num_labels=3)
    return model

def run_prediciton_url(url,  model):
    try:
        request = requests.get(url)
        with st.spinner(f'Searching for and analyzing {url}...'):
            #Get the article body
            article = Article(url)
            article.download()
            article.parse()
            body = article.text
            title = article.title
            st.header(f'Article Title : {title}')
            #Make prediction using BERT
            prediction = predict(body, model)
            #Mean sentiment
            mean = prediction.sentiment_score.mean()
            sentiment = 'negative' if mean < 0 else 'positive'
            st.subheader(f'**Overall Sentiment: {sentiment}**\n')
            st.subheader(f'**Mean score is = {mean}**')
            #Make prediction using Textblob
            blob = TextBlob(body)
            prediction['textblob_prediction'] = [sentence.sentiment.polarity for sentence in blob.sentences]
            prediction.drop('logit', axis = 1, inplace = True)

        values = st.slider('Select Sentiment Score Range', -1., 1., (-1., 1.), step = 0.01)
        st.write('Values:', values)
        prediction = prediction[(prediction['sentiment_score'] >= values[0]) & (prediction['sentiment_score'] <= values[1])]
        st.write('Prediction:')
        st.dataframe(prediction.style.\
                                highlight_max(axis=0)\
                                .background_gradient(cmap=cmap))#'RdYlGn'))
                                #.bar(subset=['sentiment_score'], align='mid', color=['#d65f5f', '#5fba7d']))
        
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

        # st.line_chart(prediction.sentiment_score)
        plt.title("Sentiment Score throughout Article")
        plt.plot(prediction.index, prediction.sentiment_score)
        plt.xlabel('Sentence Number')
        plt.axhline(y=0, linestyle = '--', color = 'black')
        plt.ylabel('Sentence Score (-1 to 1)')
        st.pyplot()

    except Exception as e:
        st.write(e)
        st.write('Web site does not exist. Please try another URL.') 


def main():
    ## SET OPTIONS
    pd.set_option('display.max_colwidth', 100)

    # Set page title
    st.title('Financial News Sentiment Analyzer')
    st.write("Welcome! This is an interactive app I created to showcase my custom sentiment analysis model for financial news.\
                You can input either raw text or a news article URL and you'll get the sentiment of every sentence in the article. The sentiment is obtained using a \
                fine-tuned BERT model on financial news articles. The performance of the model is compared \
                (side-by-side) in the table with other out-of-thebox models (in this case textblob).")
    st.markdown('[PROJECT BY NAZIH KALO](https://github.com/nazihkalo)') 
    st.write("Column definitions for the outputed table are included below.")
    # Load classification model
    with st.spinner('Loading BERT Sentiment model...'):
        model = load_model()

    st.subheader('Single Sentence classification')
    tweet_input = st.text_input('Paste Text Directly:')

    if tweet_input != '':

        # Make predictions
        with st.spinner('Predicting...'):
            prediction = predict(tweet_input, model)
            blob = TextBlob(tweet_input)
            prediction['textblob_prediction'] = [sentence.sentiment.polarity for sentence in blob.sentences]
        st.write('Prediction:')
        st.table(prediction) 

    st.subheader('Insert Financial News Article URL')

    # Get user input
    url = st.text_input('Enter News Article URL:', '')
        # As long as the query is valid (not empty)
    urls = {
        "Oil Markets":"https://www.marketwatch.com/story/us-oils-may-contract-skids-about-20-at-nadir-as-crudes-woes-continue-2020-04-19",
    "Apple": "https://finance.yahoo.com/news/apple-shares-slip-company-issues-210056567.html",
    "Airlines": "https://www.reuters.com/article/us-health-coronavirus-norwegianair/norwegian-air-gets-bondholder-deal-on-1-2-billion-debt-for-equity-swap-idUSKBN22F0LP",
    "Saudi Arabia":"https://www.reuters.com/article/health-coronavirus-saudi-finance/saudi-minister-urges-private-sector-to-ease-poor-nations-debt-burden-ft-idUSFWN2CJ14D"}
    
    url_type = st.sidebar.selectbox("Select Sample Article?",  list(urls.keys()), 0)
    url = urls[url_type]

    if url != '':
        run_prediciton_url(url, model)


    #url_type = st.sidebar.selectbox("Select Sample Article?", ['Apple Stock','Gold', 
                                                                # 'Warren Buffet Cash', "Oil Markets", 'Google'])

    # if url_type == 'Apple Stock':
    #     url = "https://finance.yahoo.com/news/apple-shares-slip-company-issues-210056567.html"
    # elif url_type == 'Apple Stock':
    #     url = "https://finance.yahoo.com/news/apple-shares-slip-company-issues-210056567.html"
    # elif url_type == 'Warren Buffet Cash':
    #     url = "https://finance.yahoo.com/news/apple-shares-slip-company-issues-210056567.html"
    # elif url_type == 'Oil Markets':
    #     url = "https://finance.yahoo.com/news/apple-shares-slip-company-issues-210056567.html"
    # elif url_type == 'Google':
    #     url = "https://finance.yahoo.com/news/apple-shares-slip-company-issues-210056567.html"


if __name__ == '__main__':
    main()
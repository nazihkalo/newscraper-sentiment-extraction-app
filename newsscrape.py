import datetime as dt
import re
import csv

import numpy as np
import pandas as pd
import streamlit as st
from pytorch_pretrained_bert.modeling import *
from pytorch_pretrained_bert.optimization import *
from bert_sentiment_utils import predict
from textblob import TextBlob
from newspaper import Article


model = BertForSequenceClassification.from_pretrained('finbert-sentiment',cache_dir=None,  num_labels=3)
from newspaper import Article
article = Article('https://www.rttnews.com/3088904/instacart-to-hire-additional-250000-full-service-shoppers.aspx?type=corp')
article.download()
article.parse()
article.text
body = article.text
prediction = predict(body, model)
blob = TextBlob(body)
prediction['textblob_prediction'] = [sentence.sentiment.polarity for sentence in blob.sentences]

print(article.text)
# Financial News Scraper + Sentiment Analysis Application

## An interactive application that takes either text or article URL as input and outputs sentiment of every sentence in the article. The sentiment is obtained using a fine-tuned BERT model on financial news articles. The performance of the model is compared (side-by-side) in the table with other out-of-thebox models (in this case textblob). 


![demo](images/appdemo.gif)
[URL used in demo](https://www.marketwatch.com/story/dow-gains-lose-altitude-and-sp-500-briefly-turns-negative-tuesday-morning-as-tech-related-and-health-care-stocks-sink-2020-04-28?mod=markets )

### Table Column Descriptions
1. Sentence: Input sentence
2. prediction: Predicted sentiment for sentence using custom model.
3. sentiment_score: Sentiment Score using custom/fine-tuned BERT model.
4. textblob_prediction: Sentiment using out of the box textblob sentiment analyzer.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from textblob import TextBlob
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
from wordcloud import WordCloud,STOPWORDS
stopwords = set(STOPWORDS)
import sentiment_lib

#####################################################################
# Parsing the entire dataset
#####################################################################
df2 = []
for dirname, _, filenames in os.walk('/Users/isabellagomez/Documents/ECE143/Final Project/archive'):
    for filename in filenames:
        if (filename.endswith('Tweets.CSV')):
            df2.append(pd.read_csv(os.path.join(dirname, filename), header=0, skiprows=lambda i: i!=0 and (i) % 50 != 0))
df = pd.concat(df2, axis=0, ignore_index=True)
df = pd.concat(df2, axis=0, ignore_index=True)
df.shape

#####################################################################
# Preprocessing the data, dropping columns
#####################################################################
tweet = df.copy()
tweet.drop(['country_code','status_id','user_id','screen_name','source','reply_to_status_id','reply_to_user_id','is_retweet','place_full_name','place_type','reply_to_screen_name','is_quote','followers_count','friends_count','account_lang','account_created_at','verified'],axis=1, inplace = True)
tweet = tweet[(tweet.lang == "en")].reset_index(drop = True)
tweet.drop(['lang'],axis=1,inplace=True)
tweet.head()

#####################################################################
# Removing Special Characters and tokens
#####################################################################
for i in range(tweet.shape[0]) :
    tweet['text'][i] = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(#[A-Za-z0-9]+)", " ", tweet['text'][i]).split()).lower()
tweet.head()

# Removing Stopwords
sentiment_lib.remove_stopwords(tweet)
sentiment_lib.remove_letter(tweet)
tweet.head()

#####################################################################
# Wordcloud of the most common words
#####################################################################
sentiment_lib.show_wordcloud(tweet['text'])

#####################################################################
# Identifying sentiment of each tweet in the dataset
#####################################################################

# Creating the tweet.sentiment column and calculating the sentiment of each tweet
sentiment_lib.sentiment_bins(tweet)
# Showing the number of tweets of each sentiment
print(tweet.sentiment.value_counts())
sns.countplot(x='sentiment', data = tweet)
plt.show()

# Showing the tendency of tweets sentiment changing over days
print("Displaying sentiment over time")
sentiment_lib.sentiment_over_time(tweet)

#####################################################################
# Calculating the frequency of most commonly occurring words in each sentiment
#####################################################################
all_words = []
all_words = [word for i in tweet.text for word in i.split() if len(word) !=1 or not isinstance(word,int)]
all_freq = sentiment_lib.get_freq(all_words)

#####################################################################
# Word cloud of the most commonly occurring positive words
#####################################################################
pos_words=sentiment_lib.count_words(tweet,'positive')
pos_freq = sentiment_lib.get_freq([word for i in pos_words for word in i.split()])
sentiment_lib.show_wordcloud(pos_words , 'POSITIVE')

#####################################################################
# Word cloud of the most commonly occurring negative words
#####################################################################
neg_words=sentiment_lib.count_words(tweet,'negative')
neg_freq = sentiment_lib.get_freq([word for i in neg_words for word in i.split()])
sentiment_lib.show_wordcloud(neg_words , 'NEGATIVE')

#####################################################################
# Word cloud of the most commonly occurring neutral words
#####################################################################
neutral_words=sentiment_lib.count_words(tweet,'neutral')
net_freq = sentiment_lib.get_freq([word for i in neutral_words for word in i.split()])
sentiment_lib.show_wordcloud(neutral_words , 'NEUTRAL')

#####################################################################
# Calculating the frequency of given keywords in each sentiment
#####################################################################
# Taking sets of keywords and calculating the number of tweets of each sentiment type that
# contain those keywords, and the frequency of each word in the keywords list as they occur
# in all the tweets
sentiment = ['positive', 'negative', 'neutral']

lockdown = ['socialdistancing', 'distancing', 'lockdown', 'stayhomestaysafe', 'quarantine', 'stayathome', 'stayhome']
freq_tweets,count_words = sentiment_lib.finding_sentiment(tweet,lockdown)
sentiment_lib.frequency_graph(lockdown, count_words, 'Count of the given keywords in the tweet dataset', xtitle="count of words", ytitle="keywords")
sentiment_lib.frequency_graph(sentiment,freq_tweets, 'Number of tweets of each sentiment containing the given keywords', "Number of tweets containing the keywords","sentiment")

#####################################################################
# Government Sentiment Analysis
#####################################################################
gov_sent_df = tweet.copy()
government_sentiments = ['government', 'trump', 'gov', 'politics', 'president', 'presidency', 'congress',
                         'senate', 'election', 'scotus', 'gop', 'potus', 'democrat', 'republican']
gov_sent_df = sentiment_lib.finding_government_sentiment(gov_sent_df, government_sentiments)
sentiment_lib.sentiment_bins(gov_sent_df)
print(gov_sent_df.sentiment.value_counts())
sns.countplot(x='sentiment', data=gov_sent_df)
plt.show()

sentiment_lib.show_wordcloud(gov_sent_df['text'])

gov_sent_df.head()

#####################################################################
# Finding Sentiment Distribution
#####################################################################
plt.figure(figsize=(10,6))
# uses polarity column to find all polarities
sns.displot(tweet['polarity'], bins=30)
plt.title('Sentiment Distribution',size = 15)
plt.xlabel('Polarity',size = 15)
plt.ylabel('Frequency',size = 15)
plt.show()

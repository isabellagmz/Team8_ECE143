
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
for dirname, _, filenames in os.walk('input'):
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

#####################################################################
# Training a CNN model to analyze the tweet sentiment
#####################################################################
wordslist = tweet['text'].apply(lambda dfs: dfs.split())
tweet_words = []
for sublist in wordslist:
    for word in sublist:
        tweet_words.append(word)
words_count = Counter(tweet_words) # calculate the frequency of each word
vocab = sorted(words_count, key=words_count.get, reverse=True)
vocab_to_int = {word:ii for ii, word in enumerate(vocab, 1)}

encoded_tweets = []
for sublist in wordslist:
    encoded_tweets.append([vocab_to_int[word] for word in sublist])
    
labels = []
for sen in tweet.sentiment:
    if sen=='negative':
        labels.append(1)
    #elif sen=='neutral':
    #   labels.append(2)
    else:
        labels.append(0)
        
tweets_len = Counter([len(sublist) for sublist in wordslist])
def pad_features(tweets, tweet_length):
    features = np.zeros((len(tweets), tweet_length), dtype=int)
    for i, row in enumerate(tweets):
        if len(row)!=0:
            features[i, :len(row):] = np.array(row)
    return features

tweet_length = 60
padded_features= pad_features(encoded_tweets, tweet_length)
print(padded_features[:2]) # show the padded features after processing

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.layers import UpSampling2D

indices = np.arange(len(encoded_tweets))
ntrain = int(len(encoded_tweets)*0.8)
ntest = len(encoded_tweets)-ntrain
# Split 80% into training sets, and 20% into test sets
x_train, x_test, y_train, y_test, train_ind, test_ind = train_test_split(padded_features, labels, indices, test_size=0.2, random_state=1)

# Scale features. Fit scaler on training only.
scaler = MinMaxScaler() #scale features between 0 and 1
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Reshape
x_train1 = x_train.reshape(ntrain,10,6,1)
x_test1 = x_test.reshape(ntest,10,6,1)

#Create labels as one-hot vectors
labels_train = tf.keras.utils.to_categorical(y_train)
labels_test = tf.keras.utils.to_categorical(y_test)

def CNN():

    model = tf.keras.models.Sequential()
    #1st hidden: Set up the first conv layer
    model.add(Conv2D(256,(3,3),activation="relu",input_shape=(10,6,1),padding='same'))
    #2nd hidden: Set up the first maxpooling layer
    model.add(MaxPooling2D(pool_size=(2,2)))
    #3rd hidden: Set up the second conv layer
    model.add(Conv2D(128,(3,3),activation="relu",padding='same'))
    #4th hidden: Set up the second maxpooling layer
    model.add(MaxPooling2D(pool_size=(2,2)))
    #5th hidden: Set up the flatten layer
    model.add(Flatten())
    #6th hidden: Set up the first dense layer
    model.add(Dense(100,activation='sigmoid'))
    #7th hidden: Set up the second dense layer
    model.add(Dense(100,activation='softmax'))
    #Output: Set up the third dense layer
    model.add(Dense(10,activation='tanh'))
    model.add(Dense(2,activation='tanh'))
    
    return model

#Compile and train the model
CNN = CNN()
CNN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(CNN.summary())
history = CNN.fit(x_train1, labels_train, epochs=20, batch_size=100, shuffle=True)
scores = CNN.evaluate(x_test1, labels_test)
print("Accuracy: %.2f%%" %(scores[1]*100))

#####################################################################
# Testing the model on a specified tweet
#####################################################################
def sentiment_cnn(tweet_text):
    encoded_text = [vocab_to_int[word] for word in tweet_text.split()]
    padded_text = np.zeros((1, tweet_length), dtype=int)
    padded_text[0,:len(encoded_text):] = np.array(encoded_text)
    padded_text = padded_text.reshape(1,10,6,1)
    l = list(CNN.predict(padded_text))
    return 'positive/neutal' if l.index(max(l))==0 else 'negative'

ind = np.random.randint(len(encoded_tweets))
tweet_text = tweet.text[ind]
print('tweet text:',tweet_text)
print('The analysis of tweet sentiment:',sentiment_cnn(tweet_text))

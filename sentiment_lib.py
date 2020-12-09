import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import os
import sentiment_lib

from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
from textblob import TextBlob
import warnings
warnings.filterwarnings("ignore")

def show_wordcloud(data,title = None):
    '''
    This function takes in a dataframe, finds the most common 200 words and returns a word cloud figure with those words

    :param data: dataframe with the text
    :param title: None or string
    :return: None
    '''

    wordcloud = WordCloud(scale=4,background_color = 'black',stopwords=stopwords,max_words=100,max_font_size=40).generate(' '.join(data))
    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    plt.title(title, size = 25)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.show()

def extract_hashtags(stoptags,df):
    '''
    This function finds the words that begin with the character '#' adds them to a new column in the dataframe
    called tags. These words represent hashtags and they are extracted to be analyzed.

    :param stoptags: list
    :param df: dataframe
    :return: None
    '''

    # check that df is a dataframe
    assert isinstance(df, pd.DataFrame)
    # check that stoptags is list
    assert type(stoptags) == list

    df['tags'] = df['text'].str.findall(r'(?:(?<=\s)|(?<=^))#.*?(?=\s|$|\.,)')
    df['tags'] = df['tags'].apply(lambda word_list:list(map(lambda w: w.lower(), word_list))).apply(lambda word_list:list(filter(lambda w: w not in stoptags, word_list)))
    
    df['accts'] = df['text'].str.findall(r'(?:(?<=\s)|(?<=^))@.*?(?=\s|$)')
    df['entity_text'] = df['tags'].apply(' '.join) + ' ' + df['accts'].apply(' '.join)
    
def remove_stopwords(df):
    '''
    This function removes 'stopwords' words(I, am, we, they, etc.) from the text in the dataframe.

    :param df: dataframe that contains text from a tweet
    :return: None
    '''

    # check that df is a dataframe
    assert isinstance(df, pd.DataFrame)

    covid_stopwords = ['covid', 'covid19', 'coronavirus', 'coronaviru']

    df['text'] = df['text'].apply(lambda dfs: ' '.join([word for word in dfs.split() if word not in stopwords and word not in covid_stopwords]))
    
def sentiment_bins(df):
    '''
    This function finds the total count of positive, negative, or neutral leaning tweets based on the
    text from the tweets

    :param df: dataframe that contains text from a tweet
    :return: None
    '''

    # check that df is a dataframe
    assert isinstance(df, pd.DataFrame)

    df['sentiment'] = ' '
    df['polarity'] = None
    for i,dfs in enumerate(df.text) :
        blob = TextBlob(dfs)
        # determine the polarity of the tweets
        df['polarity'][i] = blob.sentiment.polarity
        # assign their polarity to df column
        if blob.sentiment.polarity > 0 :
            df['sentiment'][i] = 'positive'
        elif blob.sentiment.polarity < 0 :
            df['sentiment'][i] = 'negative'
        else :
            df['sentiment'][i] = 'neutral'
    pd.set_option('display.max_colwidth', 400)

def sentiment_over_time(df):
    '''
    This function determines the change of sentiments(positive, negative, neutral) of tweets over time (15 days).

    :param df: dataframe that contains text from a tweet
    :return: None
    '''

    # check that df is a dataframe
    assert isinstance(df, pd.DataFrame)

    # format timestamp
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['created_at'] = pd.IntervalIndex(pd.cut(df['created_at'], pd.date_range('2020-03-29', '2020-05-01', freq='2880T'))).left

    df_count1 = df.groupby(['created_at','sentiment'])['text'].count().reset_index().rename(columns={'text':'count'})
    df_count1.head()
    times = df_count1.loc[df_count1['sentiment'] == 'negative']['created_at'].reset_index(drop = True)
    pos = df_count1.loc[df_count1['sentiment'] == 'positive']['count'].reset_index(drop = True)
    neutral = df_count1.loc[df_count1['sentiment'] == 'neutral']['count'].reset_index(drop = True)
    neg = df_count1.loc[df_count1['sentiment'] == 'negative']['count'].reset_index(drop = True)

    plt.figure(figsize=(10,6))
    plt.xticks(rotation='45')

    # setting titles
    plt.title("Sentiment Count vs. Time")
    plt.xlabel("Date")
    plt.ylabel("Frequency")

    # setting up legend
    lin1=plt.plot(times, pos, 'g^-', label='positive')
    lin2=plt.plot(times, neutral, 'ro-', label='neutral')
    lin3=plt.plot(times, neg, 'b--', label='negative')
    plt.legend()

    plt.show()
    
def count_words(df,sentiment):
    '''
    This function counts the number of tweets containing that sentiment in the dataframe.

    :param df: dataframe
    :param sentiment: str that says either positive, negative, or neutral
    :return: dataframe
    '''

    # check that df is a dataframe
    assert isinstance(df, pd.DataFrame)
    # check that sentiment is str
    assert type(sentiment) == str

    words = df['text'][df['sentiment'] == sentiment]
    return words

def get_freq(word_list):
    '''
    This function takes in a word_list and determines the most common 50 words in the list by counting
    their frequency.

    :param word_list: list of words from dataframe
    :return: dataframe of words and their respective frequencies
    '''

    # check that word_list is a list and not empty
    assert type(word_list) == list and len(word_list) > 0

    freq = Counter(word_list).most_common(50)
    freq = pd.DataFrame(freq)
    freq.columns = ['word', 'frequency']
    return freq

def finding_sentiment(df,keywords):
    '''
    This function finds the frequency of positive, negative, and neutral tweets in the dataframe when the
    tweets contain the keywords.

    :param df: pd.DataFrame of tweet text
    :param keywords: list of keywords to parse with
    :return: list with the counts of pos, neg, neutral tweets respectively, list of keywords
    '''

    # check that df is a dataframe
    assert isinstance(df, pd.DataFrame)
    # check that keywords is a list
    assert type(keywords) == list

    keywords_dict={key:0 for key in keywords}
    li = [0,0,0] # list with frequency of pos, neg, neutral, tweets respectively
    for data in range(len(df['text'])):
        dfs = df['text'][data]
        text1 = dfs.split()
        # if any key words are in text it processes their data
        flag = 0
        for key in keywords:
            if key in text1:
                keywords_dict[key] += 1
                flag = 1
        # adds to list a count depending on whether tweet is pos, neg, neutral
        if flag:
            if df['sentiment'][data] == 'positive':
                li[0] += 1
            elif df['sentiment'][data] == 'negative':
                li[1] += 1
            else:
                li[2] += 1
          
    return li,keywords_dict.values()

def remove_letter(df):
    '''
    This function removes any single letter words from the text in the dataframe. Is used to clean the data
    for processing the tweet information.

    :param df: dataframe that contains text from a tweet
    :return:
    '''

    # check that df is a dataframe
    assert isinstance(df, pd.DataFrame)

    df['text'] = df['text'].apply(lambda dfs: ' '.join([word for word in dfs.split() if len(word)!= 1]))
    
def frequency_graph(keyword_list,lis_name, title, xtitle, ytitle):
    '''
    This function finds the most common 50 words used in a given dataframe text and plots in bar graph
    with given titles

    :param keyword_list: list of keywords to check for
    :param lis_name: list of names for graph
    :param title: string title of the graph
    :param xtitle: string title of the x axis
    :param ytitle: string title of the y axis
    :return:
    '''

    # check that lis_name and keyword_list are lists
    assert type(keyword_list) == list
    # check that title, xtitle, ytitle are string
    assert type(title) == str and type(xtitle) == str and type(ytitle) == str
    # check that keyword_list and list names are the same length
    assert len(keyword_list) == len(lis_name)

    data = pd.DataFrame({'sentiment':lis_name,'count':keyword_list})
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.barplot(x='sentiment', y='count', 
                data=data, ax=ax)
    ax.set(xlabel=xtitle, ylabel=ytitle)
    plt.title(title)
    plt.xticks(rotation='vertical')

def finding_government_sentiment(df, keywords):
    '''
    This function returns a dataframe with column of text from tweets that refer to the government
    and covid-19.

    :param fname: name of the file importing
    :return: dataframe
    '''

    # check that df is a dataframe
    assert isinstance(df, pd.DataFrame)
    # check that keywords is a list
    assert type(keywords) == list

    data_list = []
    text = []
    for line in range(len(df['text'])):
        text.append(df['text'][line].split())
        # check if any key words are in the hashtags
        if keyword_check(text[-1], keywords):
            data_list.append(df['text'][line])

    df = pd.DataFrame(data_list, columns=['text'])
    return df

def keyword_check(hashtag_list, keywords):
    '''
    This function take a list of strings that are hashtags and check if any of the
    hashtags are in the list of keywords

    :param hashtag_list: list of strings
    :param keywords: list of strings
    :return: boolean
    '''

    # check that inputs are list
    assert type(hashtag_list) == list
    assert type(keywords) == list

    is_in = any(item in keywords for item in hashtag_list)

    return is_in

def remove_keyword(df, keywords):
    '''
    This function removes keywords from the text in the dataframe.

        :param df: dataframe that contains text from a tweet
        :return: None
        '''

    # check that df is a dataframe
    assert isinstance(df, pd.DataFrame)
    # check that keywords is a list
    assert type(keywords) == list

    df['text'] = df['text'].apply(lambda dfs: ' '.join([word for word in dfs.split() if word not in keywords]))

#!/usr/bin/env python
# coding: utf-8

# # Problem Statement
# 
# Our Objective of this problem is going to build a **Sentiment Analysis** that would help predict the stock prices. We will not be `predicting the Stock Movement - Up or Down`. 
# 
# Our clear goal would be to analyse the Sector like for e.g. we say EV so:
# 
# * How the Sector is going?
# * Why EV and Automotives?
# * Are we bullish or Bearish - This answer will come from the mass comments or statements that we will extract from the Twitter and other places. Basis this we will claim that the market is Bullish/Bearish.
# 
# * We will take 2 leading companies that have the max market share and would analyse the stock.
# * **Toughest Part is to connect the Sentiments and the Price - We will have to explore this**
# * In the End, we build a model to predict the stock prices basis two things - Some Sentiment and the Price.
# 
# * Front End will have a text Box followed by prepopulated name of companies where the user enters some text, price at which he purchased basis which a ML model runs behind the curtain and suggests that the text is bullish/bearish or positive/negative and also suggests the stock price range for 1 day, 1 week, 1 month, Quarter and Year. (This is where we will apply Confidence Interval - 99%).
# 
# Deployment Portal - Streamlit
# 
# 
# #### Quantification Part of the Project - How do you intend to make money from this?

# In[1]:


# Importing the required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px

import warnings
warnings.filterwarnings("ignore")


# yahoo finance
import yfinance as yfin


# ### Using Google News to Extract the Headlines on Electric Vehicles.

# In[ ]:


# Creating the Tag Basis Pattern
j = 99
xpath_news = '//*[@id="yDmH0d"]/c-wiz/div/div[2]/div[2]/div/main/c-wiz/div[1]/div'
xpath_news+str([j])+'/div/article/h3/a'


# In[ ]:


# Google News...
from selenium import webdriver
import time
from datetime import datetime

# Initialize the Chrome driver with the specified path
driver = webdriver.Chrome()

website = "https://news.google.com/search?pz=1&cf=all&hl=en-IN&q=Reva+Electric+Car&num=30&ict=itn2&gl=IN&ceid=IN:en"

# Open the website
driver.get(website)
driver.maximize_window()

last_height = driver.execute_script("return document.body.scrollHeight")
while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    else:
        last_height = new_height

# Perform actions with the driver as needed

# Xpath for each news item
xpath_news = '//*[@id="yDmH0d"]/c-wiz/div/div[2]/div[2]/div/main/c-wiz/div[1]/div'

# Extract headlines, text, and timestamps
gnews = []
matches = driver.find_elements(by="xpath", value=xpath_news)
for match in matches:
    for j in range(1, 101):
          # //*[@id="yDmH0d"]/c-wiz/div/div[2]/div[2]/div/main/c-wiz/div[1]/div[1]/div/article/h3/a
        # //*[@id="yDmH0d"]/c-wiz/div/div[2]/div[2]/div/main/c-wiz/div[1]/div[3]/div/article/h3/a
        # //*[@id="yDmH0d"]/c-wiz/div/div[2]/div[2]/div/main/c-wiz/div[1]/div[99]/div/article/h3/a
        # //*[@id="yDmH0d"]/c-wiz/div/div[2]/div[2]/div/main/c-wiz/div[1]/div[99]/div/article/h3/a
        # //*[@id="yDmH0d"]/c-wiz[3]/div/div[2]/div[2]/div/main/c-wiz/div[1]/div[100]/div/article/h3/a
        headline = match.find_element(by="xpath", value=xpath_news+str([j])+'/div/article/h3/a').text
        gnews.append([headline])

# Close the browser
driver.quit()

# Create a DataFrame
google = pd.DataFrame(gnews, columns=['Headline'])

# Display the DataFrame
google


# ### Downloading Maruti, Tata and M&M Share Prices

# In[2]:


maruti = yfin.Ticker("MARUTI.NS").history("max")
tata = yfin.Ticker("TATAMOTORS.NS").history("max")
mnm = yfin.Ticker("M&M.NS").history("max")


# In[4]:


# Shape of the Dataset
google.shape


# Note: In Totality, we were able to scrap the 10000 Google News Headlines from Google news on Electric Vehicles

# In[4]:


# Sentiment Analysis Libraries..

import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


# In[6]:


# Stop Words and Preprocessing the Text
stop_words = set(stopwords.words('english'))

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    
    # Remove special characters and numbers
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # Lemmatize each word in the tokens list
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join the lemmatized tokens back into a single string
    lemmatized_text = ' '.join(lemmatized_tokens)
    
    return text

google['processed_tweet'] = google['Headline'].apply(preprocess_text)


# In[7]:


# Text After Cleaning
google.head()


# In[8]:


# generate the Labels for the Model Building

from textblob import TextBlob
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

google['sentiment'] = google['processed_tweet'].apply(get_sentiment)


# In[471]:


google.sentiment.value_counts().plot(kind = "bar")


# In[87]:


# Plot the sentiment distribution
plt.figure(figsize=(8, 6))
sns.histplot(data=google, x='sentiment', bins=25, kde=True)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Distribution')
plt.show()


# ### Named Entity Recognition on the Processed Tweet

# In[9]:


# Extract named entities using NLTK

def extract_named_entities(text):
    sentences = nltk.sent_tokenize(text)
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    tagged_sentences = [nltk.pos_tag(tokens) for tokens in tokenized_sentences]
    named_entities = []
    for tagged_sentence in tagged_sentences:
        chunked_sentence = nltk.ne_chunk(tagged_sentence)
        for chunk in chunked_sentence:
            if hasattr(chunk, 'label'):
                named_entities.append(' '.join(c[0] for c in chunk))
    return named_entities

google['NE_Headline'] = google['Headline'].apply(extract_named_entities)


# In[10]:


google.head()


# In[11]:


filtered = google[google.NE_Headline.apply(lambda x: len(x)>0)]

filtered.shape


# In[12]:


# Lets remove the list from the NER Tags

def remove_list(text):
    return(' '.join(text))


# In[13]:


filtered["NE_Headline"] = filtered.NE_Headline.apply(remove_list)


# In[14]:


filtered.head()


# In[440]:


filtered.NE_Headline.unique()


# In[18]:


from dateutil import parser

def is_date(text):
    try:
        parser.parse(text)
        return True
    except ValueError:
        return False


# In[19]:


is_date("21 Mar")


# In[21]:


# Date and Time Formatting...
import dateparser
import datetime

date_list = []
month_list = []
year_list = []

list_dates = list(filtered.Time)

for item in filtered.Time:
    if isinstance(item, datetime.date):
        date_list.append(item.day)
        month_list.append(item.month)
        year_list.append(item.year)
    elif isinstance(item, str):
        date_obj = dateparser.parse(item)
        if date_obj and date_obj.year:
            date_list.append(date_obj.day)
            month_list.append(date_obj.month)
            year_list.append(date_obj.year)
        else:
            date_list.append(None)
            month_list.append(None)
            year_list.append(None)


# In[22]:


filtered["Date"] = pd.Series(date_list)
filtered["Month"] = pd.Series(month_list)
filtered["Year"] = pd.Series(year_list)


# In[23]:


filtered.head(10)


# In[24]:


filtered.isnull().sum()


# In[25]:


# Imputting Missing Values using FFill()
final = filtered.ffill()


# In[194]:


final.tail()


# In[27]:


# Get subjectivity

def Subjectivity(x):
    return(TextBlob(x).sentiment.subjectivity)

def Polarity(x):
    return(TextBlob(x).sentiment.polarity)


# In[28]:


final["Subjectivity"] = final.processed_tweet.apply(Subjectivity)

final["Polarity"] = final.processed_tweet.apply(Polarity)


# In[197]:


final.processed_tweet.value_counts()[:5]


# In[198]:


final.to_csv("FinalHeadlines.csv", index = False)


# In[31]:


modelling_data = final.drop(['Headline', "Time", "processed_tweet"], axis = 1)


# In[199]:


modelling_data.head()


# In[36]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score


# In[37]:


# Split the Data in train and test

features = modelling_data.drop(["sentiment"],axis = 1)

labels = modelling_data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(pd.get_dummies(features,drop_first = True), 
                                                    labels, 
                                                    test_size=0.2, random_state=42)


# In[38]:


# Random Forest Report
from sklearn.metrics import classification_report
rf = RandomForestClassifier()

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(report)


# In[40]:


# GBM Model

gbm = GradientBoostingClassifier()

gbm.fit(X_train, y_train)
y_pred = gbm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(report)


# In[41]:


lg = LogisticRegression()

lg.fit(X_train, y_train)
y_pred = lg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(report)


# In[42]:


# Imported Data of tata motors
tata = pd.read_csv("~/Downloads/tata.csv")


# In[43]:


tata["Year"] = tata.Date


# In[597]:


# year...
from nltk.sentiment import SentimentIntensityAnalyzer

def sentiments(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return(sentiment)


# In[614]:


sia = SentimentIntensityAnalyzer()

sia.polarity_scores("The book is lying under shelf")


# In[44]:


# get Sentiment

from nltk.sentiment import SentimentIntensityAnalyzer

def get_sentiments(text):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    return scores

neutral = []
negative = []
positive = []

for sentiment in modelling_data["sentiment"]:
    SIA = get_sentiments(sentiment)
    neutral.append(SIA['neu'])
    negative.append(SIA['neg'])
    positive.append(SIA['pos'])


# In[45]:


modelling_data["Positive"] = positive
modelling_data["Negative"] = negative
modelling_data["Neutral"] = neutral


# In[620]:


#modelled_data = modelling_data.drop("sentiment", axis = 1)


# In[206]:


modelling_data.head()


# In[49]:


tata.drop(["Date", "Volume", "Dividends", "Stock Splits"], axis = 1, inplace = True)


# In[53]:


new = pd.merge(left = modelling_data, right = tata, on = "Year", how = "left").dropna()


# In[207]:


new.head()


# In[202]:


new = new[~(new.duplicated())]


# In[204]:


new.drop(["Tomorrow", "target"], axis = 1, inplace = True)


# In[205]:


new.head()


# ## Creating Target Variable

# In[208]:


new["Diff"] = new["Close"] - new["Open"]

# sentiment is positive and diff is Pos = 1
# sentiment is negative & diff is Neg = 0

new.loc[(new.sentiment=="positive") & (new.Diff>0), "target"] = 1
new.loc[(new.sentiment=="negative") & (new.Diff<0), "target"] = 0


# In[209]:


final = new[new.target.notnull()]


# In[211]:


# Apply the Statistical Analysis to verify if NER Headline is Related to the target

import scipy.stats as stats

tbl = pd.crosstab(final.NE_Headline, final.target)
teststats, pvalue, dof, exp_freq = stats.chi2_contingency(tbl)

print(pvalue)


# In[212]:


from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, accuracy_score, classification_report

X = pd.get_dummies(final.drop("target", axis = 1), drop_first = True)
y = final.target


kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state  = 0)

lg = LogisticRegression()
predictions = []
for train_index, test_index in kfold.split(X, y):
    xtrain = X.iloc[train_index]
    ytrain = y.iloc[train_index]
    xtest = X.iloc[test_index]
    ytest = y.iloc[test_index]
    predictions.append(lg.fit(xtrain, ytrain).predict(xtest))
    
labels = pd.DataFrame(predictions).T.mode(axis = 1)[0]

print("Accuracy Score: ", accuracy_score(ytest, labels[1:]))
print("Classification Report: ")
print(classification_report(ytest, labels[1:]))
print("Cohen's Kappa Score: ", cohen_kappa_score(ytest, labels[1:]))


# In[213]:


from sklearn.ensemble import GradientBoostingClassifier

gbm = GradientBoostingClassifier()
X = pd.get_dummies(final.drop("target", axis = 1), drop_first = True)
y = final.target

predictions = []
for train_index, test_index in kfold.split(X, y):
    xtrain = X.iloc[train_index]
    ytrain = y.iloc[train_index]
    xtest = X.iloc[test_index]
    ytest = y.iloc[test_index]
    predictions.append(gbm.fit(xtrain, ytrain).predict(xtest))
    
labels = pd.DataFrame(predictions).T.mode(axis = 1)[0]

print("Accuracy Score: ", accuracy_score(ytest, labels[1:]))
print("Classification Report: ")
print(classification_report(ytest, labels[1:]))
print("Cohen's Kappa Score: ", cohen_kappa_score(ytest, labels[1:]))


# In[ ]:





# import packages
import re
import pandas as pd
import nltk
import string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
from nltk.tokenize import word_tokenize
from nltk.stem.porter import*
import nltk
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# print dataset
train=pd.read_csv("C:\\Users\\ACER\\Desktop\\tweet\\tweets_train.csv")
test=pd.read_csv("C:\\Users\\ACER\\Desktop\\tweet\\test_tweets.csv")

print(test.shape)
train.shape
train.head()
test.head()
# combime test & train dataset.
combi=train.append(test,ignore_index=True)
combi.head()
# function for remove unwanted text pattern.
def remove_pattern(input_txt,pattern):
    r=re.findall(pattern,input_txt)
    for i in r:
        input_txt=re.sub(i,'',input_txt)
        
    return input_txt
# remove twitter handles (@user)
combi['tidy_tweet']=np.vectorize(remove_pattern)(combi['tweet'],"@[\w]*")
combi.head()
# remove special characters, numbers, punctuations
combi['tidy_tweet']=combi['tidy_tweet'].str.replace("[^a-zA-Z#]","  ")
combi.head()
# remove short word having length 3 or less like oh,hmm,he etc 
combi['tidy_tweet']=combi['tidy_tweet'].apply(lambda x:'  '.join([w for w in x.split() if len(w)>3]))
combi.head()

# tokenize all the cleaned tweets i.e splitting a string of text into tokens.
tokenized_tweet=combi['tidy_tweet'].apply(lambda x: x.split())
combi['tidy_tweet_1'] = combi.apply(lambda row: word_tokenize(row['tidy_tweet']),axis=1)
combi.head()

# stripping the suffixes (“ing”, “ly”, “es”, “s” etc) from a word.
stemmer=PorterStemmer()
tokenized_tweet=tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])

# divides a text into a list of sentences
nltk.download('punkt')
combi['tidy_tweet_1']=combi['tidy_tweet_1'].apply(lambda x: [stemmer.stem(i)for i in x])
combi.head()

# function for tokenized tweet
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i]=tokenized_tweet[i]
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i]=' '.join(tokenized_tweet[i])
# combine the tokenized tweet
combi['tidy_tweet']=tokenized_tweet
combi.head()
combi['tidy_tweet'].dtypes

# visualize all the words our data using the wordcloud plot
all_words = ' '.join(''.join(txt) for txt in combi['tidy_tweet'])
wordcloud=WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(all_words)
plt.imshow(wordcloud,interpolation="bilinear")
plt.figure(figsize=(20,20))
plt.show()

racist_words = ' '.join(''.join(txt) for txt in combi['tidy_tweet'][combi['label']==1])
wordcloud=WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(racist_words)
plt.imshow(wordcloud,interpolation="bilinear")
plt.figure(figsize=(20,20))
plt.show()

non_racist_words = ' '.join(''.join(txt) for txt in combi['tidy_tweet'][combi['label']==0])
wordcloud=WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(non_racist_words)
plt.imshow(wordcloud,interpolation="bilinear")
plt.figure(figsize=(20,20))
plt.show()

# represent text into numerical features
bow_vect = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vect.fit_transform(combi['tidy_tweet'])

# model using Bag-of-Words features
train_bow=bow[:31962,:]
test_bow=bow[31962:,:]
# splitting data into training and validation set
xtrain,xvalid,ytrain,yvalid=train_test_split(train_bow,train['label'],test_size=0.3)

yvalid.dtype
lreg=LogisticRegression()
lreg.fit(xtrain,ytrain)        # training the model
prediction=lreg.predict(xvalid)                 # predicting on the validation set
prediction.dtype
f1_score(yvalid, prediction)         # calculating f1 score
# model to predict for the test data
test_pred = lreg.predict_proba(test_bow)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_pred_int
submission = test[['id','label']]
submission.to_csv('sub_lreg_bow.csv', index=False) # writing data to a CSV file

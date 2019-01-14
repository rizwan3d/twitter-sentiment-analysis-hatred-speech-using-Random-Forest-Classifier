import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as ms
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.metrics import accuracy_score

train_data = pd.read_csv('train.csv')

train_data.info()

print(train_data['label'].value_counts())

def drop_features(features,data):
    data.drop(features,inplace=True,axis=1)

# a = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ","ouch...junior is angryð#got7 #junior #yugyo..., @user")
# print(a)

def process_tweet(tweet):
    return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ",tweet.lower()).split())

train_data['processed_tweets'] = train_data['tweet'].apply(process_tweet)

print(train_data.head(10))

drop_features(['id','tweet'],train_data)

print(train_data.head(10))

train_data.info()

x_train, x_test, y_train, y_test = train_test_split(train_data["processed_tweets"],train_data["label"], test_size = 0.2, random_state = 42)

count_vect = CountVectorizer(stop_words='english')
transformer = TfidfTransformer(norm='l2',sublinear_tf=True)

x_train_counts = count_vect.fit_transform(x_train)
x_train_tfidf = transformer.fit_transform(x_train_counts)

print(x_train_counts.shape)
print(x_train_tfidf.shape)

x_test_counts = count_vect.transform(x_test)
x_test_tfidf = transformer.transform(x_test_counts)

print(x_test_counts.shape)
print(x_test_tfidf.shape)

model = RandomForestClassifier(n_estimators=200)
model.fit(x_train_tfidf,y_train)

predictions = model.predict(x_test_tfidf)
print(confusion_matrix(y_test,predictions))

f1_score(y_test,predictions)

print(accuracy_score(y_test,predictions))

#Preparing test data
test_data = pd.read_csv('test_tweets.csv')
test_data.info()
test_data['processed_tweet'] = test_data['tweet'].apply(process_tweet)
test_data.head()
drop_features(['tweet'],test_data)
train_counts = count_vect.fit_transform(train_data['processed_tweets'])
test_counts = count_vect.transform(test_data['processed_tweet'])
print(train_counts.shape)
print(test_counts.shape)
train_tfidf = transformer.fit_transform(train_counts)
test_tfidf = transformer.transform(test_counts)
print(train_tfidf.shape)
print(test_tfidf.shape)



model.fit(train_tfidf,train_data['label'])
predictions = model.predict(test_tfidf)

final_result = pd.DataFrame({'id':test_data['id'],'label':predictions})
final_result.to_csv('output.csv',index=False)
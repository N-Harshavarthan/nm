import numpy as np
import pandas as pd
import nltk
import string
import sklearn
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

'''Importing Required Modules/Packages for 
   Data Preprocessing'''

getting stopwords data
nltk.download('stopwords')


#Reading Files from given dataset(spam.csv -- kaggle)
df = pd.read_csv('spam.csv',encoding = "ISO-8859-1") 

#Removing unwanted(null) columns from dataset
df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1,inplace = True)

# print(df.isnull().sum())
# fortunately we have no other null values in available columns

#renaming the Column names 
df.rename(columns={'v1':'result','v2':'message_data'},inplace=True)

#creating the data value to store each and every elements/datatypes from dataset
data = df.where(pd.notnull(df),'')
#changing v1 column values just to <0 or 1>
data.loc[data['result']=='spam','result']=0
data.loc[data['result']=='ham','result']=1

#removing all special characters from v2 column
data['message_data'] = data['message_data'].str.replace(r'\W'," ")


#converting all uppercase strings to entirely lowercase
data['message_data']=data['message_data'].str.lower()

x=data['message_data']
y=data['result']

X_train , X_test , Y_train , Y_test = train_test_split(x,y, test_size=0.2 ,random_state= 3)

feature_extraction = TfidfVectorizer(min_df= 1, stop_words= 'english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

model = LogisticRegression()

model.fit(X_train_features,Y_train)

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test,prediction_on_test_data)

your_mail = list(map(str,input().split()))
ip_data_features = feature_extraction.transform(your_mail)
prediction = model.predict(ip_data_features)

if(prediction==1):
    print(' Not a Spam Mail')
else:
    print("Spam Mail")


print('Accuracy on Test data is about : ',accuracy_on_test_data)

print('Accuracy on Training data is about : ',accuracy_on_training_data)


#print(data) --> printing whole data
#data.to_csv('Demo spam csv.csv') --> finally converting the data to new csv file

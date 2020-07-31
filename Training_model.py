# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 18:36:04 2020

@author: keval
"""

import pandas as pd
import numpy as np

df_domain = pd.read_csv(r'C:\Users\keval\OneDrive\Desktop\Cloud_Counselage\New folder\Domain.csv')
df_event = pd.read_csv(r'C:\Users\keval\OneDrive\Desktop\Cloud_Counselage\New folder\Event.csv')

y_domain = df_domain['Domain']
y_event = df_event['Event']
X_domain = df_domain['Title']
X_event = df_event['Title']    

from sklearn.model_selection import train_test_split
X_train_domain, X_test_domain, y_train_domain, y_test_domain = train_test_split(X_domain,y_domain, test_size=0.15,random_state=11)
X_train_event, X_test_event, y_train_event, y_test_event = train_test_split(X_event,y_event, test_size=0.15,random_state=11)

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
classifier = Pipeline([('vector', CountVectorizer()),('tf', TfidfTransformer()),('classifier_sgdc', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=11)),])
classifier = classifier.fit(X_train_domain, y_train_domain)
y_pred_domain = classifier.predict(X_test_domain)


from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

results = confusion_matrix(y_test_domain, y_pred_domain)
print(results)
print('Accuracy Score for domain :',accuracy_score(y_test_domain, y_pred_domain))
print(classification_report(y_test_domain, y_pred_domain))



import pickle

# Save to file in the current working directory
pkl_filename = "domain_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(classifier, file)

# Load from file
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

from sklearn.metrics import accuracy_score     
# Calculate the accuracy score and predict target values
score = pickle_model.score(X_test_domain, y_test_domain)
print("Test score: {0:.2f} %".format(100 * score))
Ypredict = pickle_model.predict(X_test_domain)
print('Accuracy Score for domain :',accuracy_score(y_test_domain,y_pred_domain))


















from sklearn.pipeline import Pipeline
# classifier = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
classifier = Pipeline([('vector1', CountVectorizer()),('tf1', TfidfTransformer()),('classifier_sgdc1', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=11)),])
classifier = classifier.fit(X_train_event, y_train_event)


y_pred_event = classifier.predict(X_test_event)




import pickle

# Save to file in the current working directory
pkl_filename = "event_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(classifier, file)

# Load from file
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)
    
# Calculate the accuracy score and predict target values
score = pickle_model.score(X_test_event, y_test_event)
print("Test score: {0:.2f} %".format(100 * score))
Ypredict = pickle_model.predict(X_test_event)

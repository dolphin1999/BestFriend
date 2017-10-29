import pandas as pd
import numpy as np
import csv, random
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score

#nltk.download()


filenames = ['file1.txt', 'file2.txt', 'file3.txt']
with open('training1.txt', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())


df = pd.read_csv("training1.txt",sep = '\t',names = ['txt','liked'])
#df.head()

stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf = True, lowercase = True, strip_accents = 'ascii', stop_words = stopset)

y = df.liked
x = vectorizer.fit_transform(df.txt)

print y.shape
print x.shape

x_train, x_test , y_train, y_test = train_test_split(x,y,random_state = 42)

clf = naive_bayes.MultinomialNB()
clf.fit(x_train,y_train)

print roc_auc_score(y_test, clf.predict_proba(x_test)[:,1])
m = np.array(["i fell down while walking"])
m_vector=vectorizer.transform(m)
print clf.predict(m_vector)


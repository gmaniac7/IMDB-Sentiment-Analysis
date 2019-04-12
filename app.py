import os
import pickle
import re

from flask import Flask, request, jsonify

import pandas as pd
import re
import string
import numpy as np
import nltk
import nltk.classify.util
import time

import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO

# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
from nltk import ngrams
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.classify import NaiveBayesClassifier
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score
from sklearn.model_selection import train_test_split

Stop_words = set(stopwords.words('english'))

def remove_Stopwords(x):
    
    cur_ln = 0
    for protasi in x:
        protasi = [word for word in protasi.lower().split() if word not in Stop_words]
        protasi = ' '.join(protasi)
        x.loc[cur_ln] = protasi
        cur_ln+=1
    return(x)
    
def remove_Stiksis(x):
    
    cur_ln = 0
    for protasi in x:
        cleanr = re.compile('<.*?>')
        protasi = re.sub(r'\d+', '', protasi)
        protasi = re.sub(cleanr, '', protasi)
        protasi = re.sub("'", '', protasi)
        protasi = re.sub(r'\W+', ' ', protasi)
        protasi = protasi.replace('_','')
        x.iloc[cur_ln]=protasi
        cur_ln += 1
    return(x)
    
def rizes(x):
    
    arxiki = WordNetLemmatizer()
    
    cur_ln = 0
    riza = []
    for protasi in x:
        syllabes = word_tokenize(protasi)
        for leksi in syllabes:
            riza.append(arxiki.lemmatize(leksi))
        protasi = ' '.join(riza)
        x.iloc[cur_ln] = protasi
        cur_ln += 1
        riza = []
    return(x)

def no_katalikseis(x):
    
    syllabopoihsh = SnowballStemmer("english")
    
    cur_ln = 0
    riza = []
    for protasi in x:
        syllabes = word_tokenize(protasi)
        for leksi in syllabes:
            riza.append(syllabopoihsh.stem(leksi))
        protasi = ' '.join(riza)
        x.iloc[cur_ln] = protasi
        cur_ln += 1
        riza = []
    return(x)

## main function
data = pd.read_csv("imdb_master.csv",delimiter=',',encoding="latin-1")
#print(data)
data=data.drop(['Unnamed: 0', 'file'], axis=1)
data = data[:][data.label.isin(['pos','neg'])].reset_index(drop=True)

training_data = data[["review", "label"]][data.type.isin(['train'])].reset_index(drop=True)
print(training_data.head())
training_data[["label"]] = np.where(training_data[["label"]] == "pos", 1, 0)
print(training_data.head())

test_data = data[["review", "label"]][data.type.isin(['test'])].reset_index(drop=True)
test_data[["label"]] = np.where(test_data[["label"]] == "pos", 1, 0)

target_train = training_data[["label"]].values.tolist()
target_test = test_data[["label"]].values.tolist()

training_data = data[["review"]][data.type.isin(['train'])].reset_index(drop=True)
test_data = data[["review"]][data.type.isin(['test'])].reset_index(drop=True)

training_data['review'] = no_katalikseis(rizes(remove_Stopwords(remove_Stiksis(training_data['review']))))
test_data['review'] = no_katalikseis(rizes(remove_Stopwords(remove_Stiksis(test_data['review']))))
print(training_data.head())

df = pd.DataFrame(training_data[["review"]])
training_data = df["review"].tolist()

df_test = pd.DataFrame(test_data[["review"]])
#df_test = df_test.sample(1000)
test_data = df_test["review"].tolist()

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3))
tfidf_vectorizer.fit(training_data)
X = tfidf_vectorizer.transform(training_data)
X_test = tfidf_vectorizer.transform(test_data)

X_train, X_val, y_train, y_val = train_test_split(X, target_train, train_size = 0.75)


#start = time.time() Tha to xreiastoume meta
svm = LinearSVC(C=1.5)
svm.fit(X_train, y_train)
print ("Accuracy: ", round((accuracy_score(y_val, svm.predict(X_val)) * 100),2))
    
final_svm_ngram = LinearSVC(C=1.5)
final_svm_ngram.fit(X, target_train)
#end = time.time()
#dif = round((end - start),2)
print ("Final Accuracy: ", round((accuracy_score(target_test, final_svm_ngram.predict(X_test)) * 100),2))


# Unpickle the trained classifier and write preprocessor method used
def tokenizer(text):
    return text.split(' ')

def preprocessor(text):
    """ Return a cleaned version of text
    """
    # Remove HTML markup
    text = re.sub('<[^>]*>', '', text)
    # Save emoticons for later appending
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    # Remove any non-word character and append the emoticons,
    # removing the nose character for standarization. Convert to lower case
    text = (re.sub('[\W]+', ' ', text.lower()) + ' ' + ' '.join(emoticons).replace('-', ''))

    return text

#tweet_classifier = pickle.load(open('../data/logisticRegression.pkl', 'rb'))

app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    return app.send_static_file('html/index.html')


@app.route('/classify', methods=['POST'])
def classify():
    text = request.form.get('text', None)
    assert text is not None
    
    text = pd.DataFrame([text])
    text = no_katalikseis(rizes(remove_Stopwords(remove_Stiksis(text[0]))))
    print("\nModified: " + text[0] + "\n")
    X_santa = tfidf_vectorizer.transform(text)
    
    prob = final_svm_ngram.predict(X_santa[0])
    s = 'Positive' if prob == 1 else 'Negative'
    p = 1 if prob == 1 else 0
    return jsonify({
        'sentiment': s,
        'probability': p
    })

app.run()
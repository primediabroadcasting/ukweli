from flask import Flask, jsonify, request, render_template, g
from flask_cors import CORS

import pandas as pd
import gensim
import string
from nltk.tokenize import word_tokenize
import pandas as pd
import keras.preprocessing.text
import numpy as np
import feather
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pickle
import requests

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

df = pd.read_feather('cache/quotes.feather')
raw_documents = df['short'].values
gen_docs = [[w.lower() for w in word_tokenize(text)] for text in raw_documents]
dictionary = gensim.corpora.Dictionary(gen_docs)
corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
tf_idf = gensim.models.TfidfModel(corpus)
sims = gensim.similarities.MatrixSimilarity(tf_idf[corpus], num_features=len(dictionary))

model = load_model('cache/fakenews.h5')
with open('cache/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
maxlen = 256
clf = joblib.load('cache/smsspam.pkl')
vectorizer = joblib.load('cache/sms_vectorizer.pkl')

def introText(text):
    return ' '.join(keras.preprocessing.text.text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")[:25])

def preprocess(text):
    try:
        text = text.replace("' ", " ' ")
    except:
        print(text)
    signs = set(',.:;"?!')
    prods = set(text) & signs
    if not prods:
        return text

    for sign in prods:
        text = text.replace(sign, ' {} '.format(sign) )
    return text

def create_docs(df, n_gram_max=2):
    def add_ngram(q, n_gram_max):
            ngrams = []
            for n in range(2, n_gram_max+1):
                for w_index in range(len(q)-n+1):
                    ngrams.append('--'.join(q[w_index:w_index+n]))
            return q + ngrams
        
    docs = []
    for doc in df.text:
        doc = preprocess(doc).split()
        docs.append(' '.join(add_ngram(doc, n_gram_max)))
    
    return docs

def text_process(text):
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    
    return " ".join(text)

# Home page is about page
# Contains form to check
# /check POST endpoint does the actual check, but uses...
# /api to get the results
# ** update check to save files for misses on the lookup
# /admin shows lists of misses
# /admin/delete removes ones we don't care about
# /admin/add adds to the lookup, saves the file and reloads


@app.route("/")
def home():
  return render_template('home.html')

@app.route("/about/")
def about():
  return render_template('about.html')
  
@app.route("/usage/")
def usage():
  return render_template('usage.html')

@app.route("/api/", methods=["POST", "OPTIONS"])
def api():
  if request.method == 'POST':
    message = request.json['message']
    # Match on known fakes
    query_doc = [w.lower() for w in word_tokenize(introText(message))]
    query_doc_bow = dictionary.doc2bow(query_doc)
    query_doc_tf_idf = tf_idf[query_doc_bow]
    s = sims[query_doc_tf_idf]
    # match on fakenews model
    df_test = pd.DataFrame([{'text': message}])
    docs = create_docs(df_test)
    docs = tokenizer.texts_to_sequences(docs)
    docs = pad_sequences(sequences=docs, maxlen=maxlen)
    y = model.predict_proba(docs)
    print(type(s[np.argmax(s)]))
    print(type(y[0][1]))
    # match on SMS SPam model
    df_t = pd.DataFrame([{'message': message}])
    pred_feat = df_t['message'].copy()
    pred_feat = pred_feat.apply(text_process)
    pred_features = vectorizer.transform(pred_feat)
    spam = clf.predict_proba(pred_features)
    print(spam)
    data = {
      'known': round(float(s[np.argmax(s)]),2) * 100,
      'fakenews': round(float(y[0][1]),2) * 100,
      'smsspam': round(float(spam[0][1]), 2) * 100 
    }
    return jsonify(data)
  else:
    return ('', 200)

@app.route("/check", methods=['POST'])
def check():
  message = request.form['message'].strip()
  # Match on known fakes
  query_doc = [w.lower() for w in word_tokenize(introText(message))]
  query_doc_bow = dictionary.doc2bow(query_doc)
  query_doc_tf_idf = tf_idf[query_doc_bow]
  s = sims[query_doc_tf_idf]
  # match on fakenews model
  df_test = pd.DataFrame([{'text': message}])
  docs = create_docs(df_test)
  docs = tokenizer.texts_to_sequences(docs)
  docs = pad_sequences(sequences=docs, maxlen=maxlen)
  y = model.predict_proba(docs)
  print(type(s[np.argmax(s)]))
  print(type(y[0][1]))
  # match on SMS SPam model
  df_t = pd.DataFrame([{'message': message}])
  pred_feat = df_t['message'].copy()
  pred_feat = pred_feat.apply(text_process)
  pred_features = vectorizer.transform(pred_feat)
  spam = clf.predict_proba(pred_features)
  data = {
    'known': round(float(s[np.argmax(s)]),2) * 100,
    'fakenews': round(float(y[0][1]),2) * 100,
    'smsspam': round(float(spam[0][1]), 2) * 100 
  }
  return render_template('results.html', data=data, message=message)

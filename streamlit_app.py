import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

tfidf = TfidfVectorizer()
ps = PorterStemmer()

#Prepocessing

def message_transform(text):
    text =  text.lower()
    text = nltk.word_tokenize(text)  # Tokenize()
    ## Removing the special char
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    ## revoming stop words and punctuations        
    text = y[:]
    y.clear()
    for i in  text:
        if  i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    # Stemming - Reducing words to their root form to standardize them.
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidk = pickle.load(open('victorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = message_transform(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
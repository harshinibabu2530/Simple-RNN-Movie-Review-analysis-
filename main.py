import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import pad_sequences


import tensorflow as tf

#load the imdb model
word_index=imdb.get_word_index()
reversed_of_word_index={values:key for key,values in word_index.items()}

#load the h5 model
model=load_model("simple_rnn_imdb.h5",compile=False)

#helper functions
def decode_review(encoded_review):
    return " ".join([reversed_of_word_index.get(i-3,"?")for i in encoded_review])



def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padding_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padding_review


def predict_sentiment(review):
    preprocess_input=preprocess_text()
    prediction=model.predict(preprocess_input)
    sentiment='Positive' if prediction[0][0] >0.5 else 'Negative'
    return sentiment,prediction[0][0]


import streamlit as st
st.title("IMDB Movie review statement anaylysis")
st.write("Enter a movie review to classify it as positive or negative")

user_input=st.text_area("Movie Review")

if st.button("Classify"):
    preprocess_input=preprocess_text(user_input)

    predition=model.predict(preprocess_input)
    sentiment="Positive" if predition[0][0]>0.5 else "Negative"
    st.write(f"sentiment:{sentiment}")
    st.write(f"Predition_score:{predition[0][0]}")
else:
    st.write("Please enter a movie review")    
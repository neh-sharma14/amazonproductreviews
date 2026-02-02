import streamlit as st
import joblib

# 1. Load the pre-trained brain (You don't need the CSV here!)
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.title("🛒 Amazon Review Sentiment AI")

# 2. Interface
user_input = st.text_area("Paste a review to test the AI:")

if st.button("Predict"):
    if user_input:
        # Transform and Predict
        vectorized_text = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_text)
        
        if prediction[0] == 1:
            st.success("This looks like a POSITIVE review!")
        else:
            st.error("This looks like a NEGATIVE review!")

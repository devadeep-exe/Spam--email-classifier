import streamlit as st
import pickle
import re

# Load model
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# Clean function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    return text

# UI
st.title("📧 Spam Email Classifier")
st.write("Enter a message to check whether it is Spam or Not")

user_input = st.text_area("✉️ Enter your message here:")

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)

        if prediction[0] == 1:
            st.error("🚨 This is a Spam Message")
        else:
            st.success("✅ This is Not Spam")
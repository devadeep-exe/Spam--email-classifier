# import pandas as pd

# # Load dataset
# df = pd.read_csv("../data/spam.csv", encoding='latin-1')

# # Keep only required columns
# df = df[['v1', 'v2']]

# # Rename columns
# df.columns = ['label', 'message']

# # Show first 5 rows
# print(df.head())

# # Show basic info
# print("\nDataset Info:")
# print(df.info())
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    return text
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
# df = pd.read_csv("../data/spam.csv", encoding='latin-1')
df = pd.read_csv("data/spam.csv", encoding='latin-1')

# Keep required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels to numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# Convert text to numbers (TF-IDF)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)
import pickle

# Save model
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

# Predict
y_pred = model.predict(X_test_vec)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Test custom messages
while True:
    user_input = input("\nEnter a message (or type 'exit'): ")
    
    if user_input.lower() == 'exit':
        break

    input_data = vectorizer.transform([user_input])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        print("🚨 Spam Message")
    else:
        print("✅ Not Spam")
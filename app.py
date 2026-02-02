import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# 1. Load Data (Make sure you upload your 'amazon_reviews.csv' first)
df = pd.read_csv('Reviews.csv')

# 2. Text Preprocessing
# We only need the 'Text' and the 'Score' columns
X = df['Text']
y = df['Score'].apply(lambda x: 1 if x >= 4 else 0) # 1 for Good, 0 for Bad

# 3. Vectorization (Turning text into numbers)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_vec = vectorizer.fit_transform(X)

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2)

# 5. Training
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Save the outputs
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Model and Vectorizer saved!")
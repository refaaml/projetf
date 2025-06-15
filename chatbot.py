import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from wordcloud import WordCloud

# Load data correctly
df = pd.read_csv("Combined Data.csv", usecols=['statement', 'status'])
df = df.rename(columns={'statement': 'text', 'status': 'label'})

# Clean labels - assuming blank rows are "Normal"
df['label'] = df['label'].fillna('Normal')

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove numbers
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

# Vectorization with TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# WordCloud for each class
def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(title)
    plt.show()

# WordCloud for Anxiety
anxiety_text = ' '.join(df[df['label']=='Anxiety']['cleaned_text'])
generate_wordcloud(anxiety_text, 'Common Anxiety Phrases')

# WordCloud for Normal
normal_text = ' '.join(df[df['label']=='Normal']['cleaned_text'])
generate_wordcloud(normal_text, 'Common Normal Phrases')

# Prediction function
def predict_anxiety(text):
    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned])
    pred = model.predict(vect)[0]
    proba = model.predict_proba(vect)[0]
    return pred, max(proba)

# Interactive demo
print("\nAnxiety Detection Demo (type 'quit' to exit)")
while True:
    user_input = input("\nEnter a statement: ")
    if user_input.lower() == 'quit':
        break
    pred, confidence = predict_anxiety(user_input)
    print(f"Prediction: {pred} (Confidence: {confidence:.2f})")
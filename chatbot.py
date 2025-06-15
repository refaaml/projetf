import nltk
import string
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# === NLTK Setup ===
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

# === Preprocessing Function ===
def preprocess(text):
    tokens = tokenizer.tokenize(str(text).lower())
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words and t not in string.punctuation]
    return ' '.join(tokens)

# === Load Dataset ===
try:
    df = pd.read_csv("Combined Data.csv", usecols=['statement', 'status'], encoding='latin-1')
    df.rename(columns={'statement': 'response_text', 'status': 'class'}, inplace=True)
    print("\n✅ Data Loaded:")
    print(df.head())
except FileNotFoundError:
    print("❌ Fichier Combined Data.csv non trouvé.")
    exit()

# === Cleaning & Vectorization ===
df.dropna(subset=['response_text', 'class'], inplace=True)
df['cleaned_text'] = df['response_text'].apply(preprocess)
le = LabelEncoder()
df['label'] = le.fit_transform(df['class'])

X = df['cleaned_text']
y = df['label']

vectorizer = CountVectorizer()
X_vect = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.3, random_state=42)

# === Decision Tree Hyperparameter Tuning ===
param_dt = {
    'max_depth': [None, 10, 20, 30],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 5, 10]
}
grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_dt, cv=5, scoring='accuracy')
grid_dt.fit(X_train, y_train)
best_dt = grid_dt.best_estimator_
y_pred_dt = best_dt.predict(X_test)

print("\n=== 📊 Decision Tree Report ===")
print("Best Params:", grid_dt.best_params_)
print(classification_report(y_test, y_pred_dt, target_names=le.classes_))
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.2f}")

# === Naive Bayes Hyperparameter Tuning ===
param_nb = {
    'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],
    'fit_prior': [True, False]
}
grid_nb = GridSearchCV(MultinomialNB(), param_nb, cv=5, scoring='accuracy')
grid_nb.fit(X_train, y_train)
best_nb = grid_nb.best_estimator_
y_pred_nb = best_nb.predict(X_test)

print("\n=== 📊 Naive Bayes Report ===")
print("Best Params:", grid_nb.best_params_)
print(classification_report(y_test, y_pred_nb, target_names=le.classes_))
print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.2f}")

# === WordCloud (optional) ===
def show_wordcloud():
    text = " ".join(df['response_text'].astype(str))
    wordcloud = WordCloud(background_color="white", stopwords=stop_words).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("WordCloud des réponses")
    plt.show()

# === Classify New Input with Decision Tree (default) ===
def classify_input(text, model=best_dt):
    cleaned = preprocess(text)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)[0]
    return le.inverse_transform([prediction])[0]

# === Chatbot Console ===
def chatbot():
    print("\n🤖 Bonjour ! Tapez 'exit' pour quitter.")
    while True:
        user_input = input("Vous: ")
        if user_input.lower() == 'exit':
            print("Bot: Au revoir 👋")
            break
        category = classify_input(user_input)
        print(f"Bot: '{category}'\n")

# === Run All ===
if __name__ == "__main__":
    show_wordcloud()
    chatbot()

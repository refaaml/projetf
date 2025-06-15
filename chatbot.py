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
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# === Préparation nltk ===
nltk.download('stopwords')

# === Prétraitement ===
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

# Fonction de nettoyage du texte
def preprocess(text):
    tokens = tokenizer.tokenize(text.lower())
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words and t not in string.punctuation]
    return ' '.join(tokens)

# === Chargement des données ===
try:
    df = pd.read_csv("Combined Data.csv", usecols=['response_id', 'class', 'response_text'], encoding='latin-1')
    print("\nAperçu des données:")
    print(df.head())
except FileNotFoundError:
    print("Fichier Combined Data.csv non trouvé. Utilisation de données par défaut.")
    

# === Nettoyage et vectorisation ===
df.dropna(subset=['response_text', 'class'], inplace=True)
df['cleaned_text'] = df['response_text'].apply(preprocess)
le = LabelEncoder()
df['label'] = le.fit_transform(df['class'])

X = df['cleaned_text']
y = df['label']

vectorizer = CountVectorizer()
X_vect = vectorizer.fit_transform(X)

# === Division des données ===
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.3, random_state=42)

# === Entraînement du modèle ===
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# === Évaluation ===
y_pred = dt_model.predict(X_test)
print("\n=== Évaluation de l'arbre de décision ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# === WordCloud (optionnel) ===
def show_wordcloud():
    text = " ".join(df['response_text'].astype(str))
    wordcloud = WordCloud(background_color="white", stopwords=stop_words).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("WordCloud des réponses")
    plt.show()

# === Classification d'un texte ===
def classify_input(text):
    cleaned = preprocess(text)
    vect = vectorizer.transform([cleaned])
    prediction = dt_model.predict(vect)[0]
    return le.inverse_transform([prediction])[0]

# === Chatbot Console ===
def chatbot():
    print("\n🤖 Bonjour ! Je peux vous aider avec des questions de support technique, facturation ou livraison.")
    print("Tapez 'exit' pour quitter.\n")
    while True:
        user_input = input("Vous: ")
        if user_input.lower() == 'exit':
            print("Bot: Au revoir ! 👋")
            break
        category = classify_input(user_input)
        print(f"Bot (Arbre de décision): '{category}'\n")

# === Exécution ===
if __name__ == "__main__":
    show_wordcloud()
    chatbot()
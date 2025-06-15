import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils import class_weight

print("Downloading NLTK stopwords (if not already downloaded)...")
nltk.download('stopwords')

# === 1. Load and clean dataset ===
print("Loading dataset...")
df = pd.read_csv("Combined Data.csv", usecols=['statement', 'status'])
df = df.rename(columns={'statement': 'text', 'status': 'label'})
df['label'] = df['label'].fillna('Normal')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\d+', '', text)      # remove digits
    return text

print("Cleaning text data...")
df['cleaned_text'] = df['text'].apply(clean_text)

# === 2. Vectorization ===
print("Vectorizing text...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']

# === 3. Train/Test split ===
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === 4. Train and tune Decision Tree ===
print("Training Decision Tree with GridSearchCV...")
dt = DecisionTreeClassifier(random_state=42)
param_grid = {
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid = GridSearchCV(dt, param_grid, cv=5)
grid.fit(X_train, y_train)
best_dt = grid.best_estimator_

# Handle class imbalance with sample weights
weights = class_weight.compute_sample_weight('balanced', y_train)
print("Fitting best model with sample weights...")
best_dt.fit(X_train, y_train, sample_weight=weights)

# === 5. Evaluate model ===
print("\nEvaluating model on test set...")
y_pred = best_dt.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# === 6. Prediction demo ===
def predict_text(text):
    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned])
    pred = best_dt.predict(vect)[0]
    conf = best_dt.predict_proba(vect)[0].max() if hasattr(best_dt, "predict_proba") else None
    return pred, conf

print("\n=== Prediction demo ===")
test_statements = [
    "I feel nervous all the time.",
    "Everything is going well in my life.",
    "Sometimes I can't breathe when I'm scared.",
    "I'm very confident today.",
    "I'm losing interest in everything."
]

for i, statement in enumerate(test_statements, 1):
    prediction, confidence = predict_text(statement)
    if confidence:
        print(f"{i}. '{statement}'\n   ➤ Prediction: {prediction} (confidence: {confidence:.2f})")
    else:
        print(f"{i}. '{statement}'\n   ➤ Prediction: {prediction}")

print("\nScript finished successfully!")

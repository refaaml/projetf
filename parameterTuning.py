#  Import required libraries
import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# 📊 Bokeh libraries for visualization
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource

# 📁 Save output as HTML file
output_file("results.html")

# 🧹 Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

# 📥 Load the dataset
data = pd.read_csv("Combined Data.csv", usecols=["response_text", "class"], encoding='latin-1')
data = data.dropna()

# 🧽 Clean the text column
data["clean_text"] = data["response_text"].apply(clean_text)

# 🎯 Define features and labels
X = data["clean_text"]
y = data["class"]

# 🔢 Convert text to numerical vectors using TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=1000)
X_vec = vectorizer.fit_transform(X)

# 🔀 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# 🤖 1. Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)
nb_acc = accuracy_score(y_test, nb_preds)

# 🌳 2. Decision Tree with different depths
depths = [3, 5, 7, 10, 15]
best_acc = 0
best_model = None
best_preds = None
best_depth = 0
acc_list = []

for depth in depths:
    dt_model = DecisionTreeClassifier(max_depth=depth)
    dt_model.fit(X_train, y_train)
    dt_preds = dt_model.predict(X_test)
    acc = accuracy_score(y_test, dt_preds)
    acc_list.append(acc)
    print( acc_list)
    
    if acc > best_acc:
        best_acc = acc
        best_model = dt_model
        best_preds = dt_preds
        best_depth = depth

# ✅ Use best_preds for confusion matrix and evaluation
print("\n===== Naive Bayes Classifier =====")
print("Accuracy:", nb_acc)
print(classification_report(y_test, nb_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, nb_preds))

print(f"\n===== Best Decision Tree Classifier (depth={best_depth}) =====")
print("Accuracy:", best_acc)
print(classification_report(y_test, best_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, best_preds))

# 📈 Bokeh Visualizations

# 1. Accuracy Comparison Plot
models = ['Naive Bayes'] + [f'DT_depth_{d}' for d in depths]
accuracies = [nb_acc] + acc_list
source_acc = ColumnDataSource(data={'Model': models, 'Accuracy': accuracies})

p1 = figure(x_range=models, title="Model Accuracies", height=300)
p1.vbar(x='Model', top='Accuracy', width=0.6, source=source_acc, color="skyblue")
p1.xaxis.major_label_orientation = 1
p1.y_range.start = 0
p1.yaxis.axis_label = "Accuracy"

# 2. Confusion Matrix for Naive Bayes
cm_nb = confusion_matrix(y_test, nb_preds)
labels = [str(i) for i in range(len(cm_nb))]
source_cm_nb = ColumnDataSource(data={'x': labels*len(labels), 'y': sorted(labels*len(labels), reverse=True), 'value': cm_nb.flatten()})

p2 = figure(title="Naive Bayes Confusion Matrix", x_range=labels, y_range=list(reversed(labels)))
p2.rect(x='x', y='y', width=1, height=1, source=source_cm_nb, fill_color='orange', line_color='black')
p2.xaxis.axis_label = "Predicted"
p2.yaxis.axis_label = "Actual"

# 3. Confusion Matrix for Best Decision Tree
cm_dt = confusion_matrix(y_test, best_preds)
source_cm_dt = ColumnDataSource(data={'x': labels*len(labels), 'y': sorted(labels*len(labels), reverse=True), 'value': cm_dt.flatten()})

p3 = figure(title=f"Decision Tree (depth={best_depth}) Confusion Matrix", x_range=labels, y_range=list(reversed(labels)))
p3.rect(x='x', y='y', width=1, height=1, source=source_cm_dt, fill_color='green', line_color='black')
p3.xaxis.axis_label = "Predicted"
p3.yaxis.axis_label = "Actual"

# 4. Accuracy by Depth (only for DT)
source_dt = ColumnDataSource(data={'depth': depths, 'accuracy': acc_list})
p4 = figure(title="Decision Tree Accuracy by Depth", x_axis_label="Depth", y_axis_label="Accuracy")
p4.line('depth', 'accuracy', source=source_dt, line_width=2, color="blue")
p4.circle('depth', 'accuracy', source=source_dt, size=8, color="red")

# 📊 Show all plots in HTML
show(column(p1, row(p2, p3), p4))
# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import gensim.downloader as api
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import os


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)
CORS(app) 


print("Loading Word2Vec model")
model = api.load("word2vec-google-news-300")  


stop_words = set(stopwords.words('english'))
negation_words = ['not', "don't", 'no', 'never', "can't", "won't"]

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = []
    negation = False
    for word in tokens:
        if word in negation_words:
            negation = not negation  
            continue  
        if word.isalpha() and word not in stop_words:
            if negation:
                word = 'not_' + word  
            filtered_tokens.append(word)
            negation = False  
    return filtered_tokens

def get_sentence_vector(tokens):
    vectors = []
    for word in tokens:
        negate = False
        if word.startswith('not_'):
            word = word[4:]  
            negate = True
        try:
            vec = model[word]
            if negate:
                vec = -vec  
            vectors.append(vec)
        except KeyError:
            continue 
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


print("Loading dataset")
data = pd.read_csv("emotions.csv")


X = []
y = []
print("Processing text data")
for idx, row in data.iterrows():
    sentence, label = row['text'], row['label']
    tokens = preprocess_text(sentence)
    vector = get_sentence_vector(tokens)
    X.append(vector)
    y.append(label)


X = np.array(X)
y = np.array(y)


print("Applying SMOTE to balance the dataset")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


print("Splitting data into training and testing sets")
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)


print("Training Logistic Regression classifier")
classifier = LogisticRegression(
    max_iter=1000, multi_class='multinomial', solver='lbfgs', class_weight='balanced'
)
classifier.fit(X_train, y_train)


print("Predicting on test data")
y_pred = classifier.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


numeric_to_string_mapping = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=[numeric_to_string_mapping[i] for i in range(6)]))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_input = data.get('text', '')
    tokens = preprocess_text(user_input)
    vector = get_sentence_vector(tokens).reshape(1, -1)
    predicted_class = classifier.predict(vector)[0]
    emotion = numeric_to_string_mapping.get(predicted_class, 'unknown')
    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

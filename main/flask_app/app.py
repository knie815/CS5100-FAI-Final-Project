# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import gensim
from gensim.models import KeyedVectors
import gensim.downloader as api
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

app = Flask(__name__)
CORS(app)  


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



data = pd.read_csv("emotions.csv")
train_data = list(zip(data["text"], data["label"]))

numeric_to_string_mapping = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

label_mapping = {label: idx for idx, label in enumerate(numeric_to_string_mapping.values())}

train_data = [(sentence, numeric_to_string_mapping[label]) for sentence, label in train_data]

X = []
y = []

for sentence, label in train_data:
    tokens = preprocess_text(sentence)
    vector = get_sentence_vector(tokens)
    X.append(vector)
    y.append(label_mapping[label])


X = np.array(X)
y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



mean_vectors = []
for cl in np.unique(y_train):
    mean_vectors.append(np.mean(X_train[y_train == cl], axis=0))


S_W = np.zeros((model.vector_size, model.vector_size))
for cl, mv in zip(np.unique(y_train), mean_vectors):
    class_scatter = np.zeros((model.vector_size, model.vector_size))
    for row in X_train[y_train == cl]:
        row, mv = row.reshape(model.vector_size, 1), mv.reshape(model.vector_size, 1)
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter


overall_mean = np.mean(X_train, axis=0).reshape(model.vector_size, 1)
S_B = np.zeros((model.vector_size, model.vector_size))
for i, mean_vec in enumerate(mean_vectors):
    n = X_train[y_train == i, :].shape[0]
    mean_vec = mean_vec.reshape(model.vector_size, 1)
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)


eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(S_W).dot(S_B))


eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]


eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)


k = len(np.unique(y_train)) - 1  
W = np.hstack([eig_pairs[i][1].reshape(model.vector_size, 1) for i in range(k)])


X_train_lda = X_train.dot(W)
X_test_lda = X_test.dot(W)


mean_vectors_lda = []
for cl in np.unique(y_train):
    mean_vectors_lda.append(np.mean(X_train_lda[y_train == cl], axis=0))

y_pred = []
for sample in X_test_lda:
    distances = [np.linalg.norm(sample - mean_vec) for mean_vec in mean_vectors_lda]
    y_pred.append(np.argmin(distances))


accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_input = data.get('text', '')
    tokens = preprocess_text(user_input)
    vector = get_sentence_vector(tokens)
    vector = vector.reshape(1, -1)
    vector_lda = vector.dot(W)
    distances = []
    for mean_vec in mean_vectors_lda:
        distances.append(np.linalg.norm(vector_lda - mean_vec))
    predicted_class = np.argmin(distances)
    emotion = numeric_to_string_mapping[predicted_class]
    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True)

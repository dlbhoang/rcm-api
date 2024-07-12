from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import warnings
from helpers import vn_processing as xt

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load trained models and vectorizer
with open('models/xgboots_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Function to preprocess text and make predictions
def predict_sentiment(text):
    df = pd.DataFrame({'Comment': text})
    df['Comment Tokenize'] = df['Comment'].apply(xt.stepByStep)
    X_test = tfidf.transform(df['Comment Tokenize'])
    y_pred = model.predict(X_test)
    df['Label'] = y_pred
    df['Label'] = df['Label'].map({0: 'Negative', 1: 'Neutral', 2: 'Positive'})
    df = df[['Comment', 'Label']]
    return df.to_dict(orient='records')

@app.route('/')
def index():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form['text']
        result = predict_sentiment([text])
        return render_template('index.html', result=result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

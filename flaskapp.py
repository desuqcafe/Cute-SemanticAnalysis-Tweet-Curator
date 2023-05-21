from flask import Flask, request, render_template, jsonify
import requests
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import configparser
import os

config = configparser.ConfigParser()
config_path = 'C:/Users/desuq/luminavault/semantic_flask_app/config.ini'

try:
    if not os.path.exists(config_path):
        raise ValueError("Config file doesn't exist")
    config.read(config_path)
    
    CLIENT_ID = config.get('Credentials', 'CLIENT_ID')
    SECRET_KEY = config.get('Credentials', 'SECRET_KEY')
    USERNAME = config.get('Credentials', 'USERNAME')
    PASSWORD = config.get('Credentials', 'PASSWORD')

    #print(CLIENT_ID)
    #print(SECRET_KEY)
    #print(USERNAME)
    #print(PASSWORD)

except ValueError as ve:
    print(ve)
except configparser.NoOptionError as noe:
    print("Error in config file: ", noe)
except Exception as e:
    print("Unexpected error: ", e)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\W', ' ', text)
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

app = Flask(__name__)

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# authenticate
auth = requests.auth.HTTPBasicAuth(CLIENT_ID, SECRET_KEY)
data = {
    'grant_type': 'password',
    'username': USERNAME,
    'password': PASSWORD
}
headers = {'User-Agent': 'MYAPI/0.0.1'}
res = requests.post('https://www.reddit.com/api/v1/access_token', auth=auth, data=data, headers=headers)
TOKEN = res.json()['access_token']

# update headers with the token
headers['Authorization'] = f'Bearer {TOKEN}'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['tweet']
        preprocessed_text = preprocess_text(text)
        features = vectorizer.transform([preprocessed_text])
        prediction = model.predict(features)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        return render_template('result.html', sentiment=sentiment, tweet=text) # Pass the tweet text to the template
    
@app.route('/get_sentiment', methods=['POST'])
def get_sentiment():
    text = request.form['text']
    preprocessed_text = preprocess_text(text)
    features = vectorizer.transform([preprocessed_text])
    prediction = model.predict(features)
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    return jsonify({'sentiment': sentiment})

@app.route('/search_user', methods=['POST'])
def search_user():
    user_to_search = request.form['username']
    # get the last 5 posts of the user
    response = requests.get(f'https://oauth.reddit.com/user/{user_to_search}/submitted?limit=5', headers=headers)
    # get the titles and content of the last 5 posts
    posts = [{'title': post['data']['title'], 'content': post['data']['selftext']} for post in response.json()['data']['children']]
    return jsonify(posts)

if __name__ == '__main__':
    app.run(debug=True)
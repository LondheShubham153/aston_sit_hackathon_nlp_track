from flask import Flask,jsonify,request
import pickle
from sentiment_analysis import SentimentAnalyzer
from preprocess_tweets import PreProcessor
analyzer = SentimentAnalyzer()

app = Flask(__name__)

@app.route('/analyse', methods=['POST'])
def welcome():
    request_tweet = request.json["data"]
    filename = "models/finalized_model.sav"
    loaded_model = pickle.load(open(filename, 'rb'))
    sentiment = analyzer.analyse(loaded_model,request_tweet)
    return jsonify({"Sentiment": sentiment})

if __name__ == '__main__':
    app.run()
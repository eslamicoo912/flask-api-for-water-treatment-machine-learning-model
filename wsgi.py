from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pickle
import joblib

app = Flask(__name__)

CORS(app)

with open('./model/multi-water-quality.pkl', 'rb') as file:
    model = joblib.load(file)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get the JSON data from the request
    print(data)
    # Extract the features from the JSON data
    features = np.array(list(data.values())).reshape(1, -1)

    # Make predictions using the loaded model
    prediction = model.predict(features)

    # Convert the prediction to a human-readable format
    print('prediction: ', prediction[0][0])
    if prediction[0][0] > 2.8 or prediction[0][1] > 0.005:
        result = "Not Good"
    else:
        result = "Good"

    # Return the prediction as a JSON response
    return jsonify({'prediction': result})


@app.route("/")
def running():
    return 'Flask is running now'


if __name__ == "__main__":
    app.run()

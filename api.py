#%%
"""Simple Flask API for a sklearn model

This script is used to start a flask application to call a sklearn model

Usage:
    To start the server, use: `python3 api.py`
    To do prediction, call the end point with data
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np

#%% Setup
app = Flask(__name__)

#%%
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    data = np.array([data])
    result = model.predict(data)
    output = {}
    output["code"] = "success"
    output["predict_class"] = str(result[0])
    print(output)
    return output


if __name__ == "__main__":
    model = joblib.load("model.pkl")
    app.run(host="0.0.0.0", port=8080, debug=True)

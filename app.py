import joblib
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
# Load all components using joblib
rfmodel = joblib.load('model1_rf.joblib')
scalar = joblib.load('scaler.joblib')
encoder = joblib.load('encoder.joblib') 

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data=request.json['data']
    df = pd.DataFrame([data])
    #df['category_column'] = encoder.transform(df['category_column'])
    scaled_data = scalar.transform(df)
    prediction = rfmodel.predict(scaled_data)
    result = int(prediction[0])
    print(f"Predicted Incident Priority/Category: {result}")

    return jsonify(result)

if __name__=="__main__":
    app.run(debug=True)

import joblib
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
# Load all components using joblib
rfmodel = joblib.load('model1_rf.joblib')
scalar = joblib.load('scaler.joblib')
encoder = joblib.load('label_encoder.pkl') 

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
    pred_val = int(prediction[0])
    result_word = encoder.inverse_transform([pred_val])[0]
    print(f"Predicted Incident Priority/Category: {result_word}")
    return jsonify({'priority': result_word, 'category_id':pred_val})

@app.route('/predict', methods=['POST'])
def predict():
  
    features = [
        float(request.form['Impact']),
        float(request.form['Urgency']),
        float(request.form['Incident Category']),
        float(request.form['Location']),
        float(request.form['Service']),
        float(request.form['user_group'])
        ]
    final_input = np.array(features).reshape(1,-1)
    scaled_input = scalar.transform(final_input)
    prediction = rfmodel.predict(scaled_input)
    pred_val = int(prediction[0])
    if pred_val<len(encoder.classes_):
        result_word=encoder.inverse_transform([pred_val])[0]
    else:
        result_word=f"Unknown Priority Level (Value:{pred_val})"    
    return render_template("home.html", prediction_text = f"The Predicted Incident Priority:{result_word}")

if __name__=="__main__":
    app.run(debug=True)
    


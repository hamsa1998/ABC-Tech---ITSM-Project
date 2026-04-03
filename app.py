import joblib
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
# Load all components using joblib
rfmodel = joblib.load('model1_rf.joblib')
scalar = joblib.load('scaler.joblib')
encoder = joblib.load('label_encoder.pkl')
wbs_encoder = joblib.load('wbs_encoder.pkl')

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
    try:
        ci_cat = float(request.form['CI_Cat'])
        ci_subcat = float(request.form['CI_Subcat'])
        category = float(request.form['Category'])
        wbs_input = request.form['WBS'].strip()

        # WBS mapping fallback for unknown inputs
        try:
            wbs_encoded = int(wbs_encoder.transform([wbs_input])[0])
        except ValueError:
            wbs_encoded = None
            if wbs_input.isdigit():
                # try as WBS0000xx style if user provides number
                wbs_try = 'WBS' + wbs_input.zfill(6)
                if wbs_try in wbs_encoder.classes_:
                    wbs_encoded = int(wbs_encoder.transform([wbs_try])[0])
            if wbs_encoded is None:
                # if still not found, pick the closest known WBS class by index
                all_classes = list(wbs_encoder.classes_)
                if len(all_classes) > 0:
                    wbs_encoded = int(wbs_encoder.transform([all_classes[0]])[0])
                else:
                    return render_template('home.html', prediction_text='Error: WBS encoder has no classes')

        features = [ci_cat, ci_subcat, wbs_encoded, category]
        final_input = np.array(features).reshape(1, -1)
        scaled_input = scalar.transform(final_input)
        prediction = rfmodel.predict(scaled_input)
        pred_val = int(prediction[0])

        # Convert priority number to a human label.
        # Adjust thresholds as needed according to your actual priority scale.
        if pred_val in [1, 2]:
            priority_label = 'Low'
        elif pred_val == 3:
            priority_label = 'Medium'
        else:
            priority_label = 'High'

        return render_template('home.html', prediction_text=f"Predicted Incident Priority: {priority_label} (raw {pred_val})")

    except Exception as e:
        return render_template('home.html', prediction_text=f"Error during prediction: {e}")

if __name__=="__main__":
    app.run(debug=True)
    


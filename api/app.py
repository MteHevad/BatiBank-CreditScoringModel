from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('credit_risk_model.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the POST request
    data = request.json
    df = pd.DataFrame(data, index=[0])
    
    # Make prediction
    prediction = model.predict(df)
    
    # Return the result as JSON
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)

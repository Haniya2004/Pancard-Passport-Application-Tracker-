from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Predict status function
def predict_status(days_since_submission, id_proof, application_type):
    # Create a dataframe for the input features with the same column names and order as training data
    input_data = pd.DataFrame({
        'Days_Since_Submission': [days_since_submission],
        'ID_Proof': [id_proof],  # Assumed to be already encoded (e.g., 0 or 1)
        'Application_Type': [application_type]  # Assumed to be already encoded (e.g., 0 or 1)
    })

    # Predict the status using the loaded model
    predicted_status = model.predict(input_data)

    return predicted_status[0]  # Returning encoded status (e.g., 0 for "Pending" and 1 for "Approved")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    application_id = data['application_id']
    days_since_submission = int(data['days_since_submission'])
    id_proof = int(data['id_proof'])  # Ensure input matches encoded values (0 or 1)
    application_type = int(data['application_type'])  # Ensure input matches encoded values (0 or 1)

    # Predict the status using the function
    status = predict_status(days_since_submission, id_proof, application_type)

    # Return JSON response with status and application ID
    return jsonify({'application_id': application_id, 'status': status})

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        application_id = data['application_id']
        days_since_submission = int(data['days_since_submission'])
        id_proof = int(data['id_proof'])
        application_type = int(data['application_type'])

        status = predict_status(days_since_submission, id_proof, application_type)
        print(f"Prediction Input: {days_since_submission}, {id_proof}, {application_type}")
        print(f"Predicted Status: {status}")
        return jsonify({'application_id': application_id, 'status': status})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)})

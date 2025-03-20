from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained pipeline
pipeline = joblib.load('model_pipeline.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        "Age": request.form.get("age"),
        "Processing Time": request.form.get("processing_time"),
        "Escalation Count": request.form.get("escalation_count"),
        "Issue Resolution Time": request.form.get("issue_resolution_time"),
        "Occupation Type": request.form.get("occupation_type"),
        "Submission Channel": request.form.get("submission_channel"),
        "Document Verification Status": request.form.get("document_verification_status"),
        "ID Proof Type": request.form.get("id_proof_type"),
        "Address Proof Type": request.form.get("address_proof_type"),
        "Assigned Officer/Branch": request.form.get("assigned_officer_branch")
    }

    input_df = pd.DataFrame([data])
    prediction = pipeline.predict(input_df)
    return jsonify({"Prediction": prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)

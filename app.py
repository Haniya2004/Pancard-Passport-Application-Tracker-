from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load the modified dataset
df = pd.read_csv('modified_pan.csv')

# Encode categorical variables
id_proof_encoder = LabelEncoder()
status_encoder = LabelEncoder()
application_type_encoder = LabelEncoder()

df['ID_Proof'] = id_proof_encoder.fit_transform(df['ID_Proof'])
df['Status'] = status_encoder.fit_transform(df['Status'])
df['Application_Type'] = application_type_encoder.fit_transform(df['Application_Type'])

# Split the data into features and target variable
X = df.drop(columns=['Status'])
y = df['Status']

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Function to predict status based on input features
def predict_status(days_since_submission, id_proof, application_type):
    # Create a dataframe for the input features with the same column names and order as training data
    input_data = pd.DataFrame({
        'Days_Since_Submission': [days_since_submission],
        'ID_Proof': [id_proof],
        'Application_Type': [application_type]
    }, columns=X.columns)  # Ensure the columns match the training data
    
    # Predict the status
    predicted_status = model.predict(input_data)
    
    # Decode the predicted status
    predicted_status = status_encoder.inverse_transform(predicted_status)
    
    return predicted_status[0]

# Flask web application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    days_since_submission = data['days_since_submission']
    id_proof = id_proof_encoder.transform([data['id_proof']])[0]
    application_type = application_type_encoder.transform([data['application_type']])[0]
    
    prediction = predict_status(days_since_submission, id_proof, application_type)
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
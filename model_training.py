import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df = pd.read_csv('pan.csv')

# Convert date columns to datetime format
df['Date_of_Birth'] = pd.to_datetime(df['Date_of_Birth'], format='%d-%m-%Y')
df['Submission_Date'] = pd.to_datetime(df['Submission_Date'], format='%d-%m-%Y')

# Calculate Days_Since_Submission
df['Days_Since_Submission'] = (pd.to_datetime('2025-03-22') - df['Submission_Date']).dt.days

# Encode categorical variables
id_proof_encoder = LabelEncoder()
status_encoder = LabelEncoder()
application_type_encoder = LabelEncoder()

df['ID_Proof'] = id_proof_encoder.fit_transform(df['ID_Proof'])
df['Status'] = status_encoder.fit_transform(df['Status'])
df['Application_Type'] = application_type_encoder.fit_transform(df['Application_Type'])

# Drop original date columns, Date_of_Birth column, Place_of_Birth column, and Application_ID column
df = df.drop(columns=['Date_of_Birth', 'Submission_Date', 'Place_of_Birth', 'Application_ID'])

# Split the data into features and target variable
X = df.drop(columns=['Status'])
y = df['Status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict the status on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Function to predict status based on input features
def predict_status(days_since_submission, id_proof, application_type):
    # Create a dataframe for the input features with the same column names and order as training data
    input_data = pd.DataFrame({
        'Days_Since_Submission': [days_since_submission],
        'ID_Proof': [id_proof],
        'Application_Type': [application_type]
    }, columns=X_train.columns)  # Ensure the columns match the training data
    
    # Predict the status
    predicted_status = model.predict(input_data)
    
    # Decode the predicted status
    predicted_status = status_encoder.inverse_transform(predicted_status)
    
    return predicted_status[0]


# Example usage of predict_status function with values from the dataset to avoid KeyError
example_prediction = predict_status(200, id_proof_encoder.transform(['aadhar card'])[0], application_type_encoder.transform(['New'])[0])
print(f"Predicted Status: {example_prediction}")

# Save the modified dataset to a new CSV file
df.to_csv('modified_pan.csv', index=False)

# Save the trained model and encoder to files
joblib.dump(model, 'model.pkl')  # Save the model
joblib.dump(status_encoder, 'status_encoder.pkl')  # Save the encoder for decoding predictions later
print("Model and encoder have been trained and saved.")
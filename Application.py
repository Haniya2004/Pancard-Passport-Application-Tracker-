import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import tkinter as tk
from tkinter import messagebox

# Load the cleaned dataset
df = pd.read_csv('Pancard_dataset.csv')

# Convert Date of Birth column to datetime format
df['Date of Birth'] = pd.to_datetime(df['Date of Birth'], format='%d-%m-%Y')

# Feature Engineering: Create age from Date of Birth
df['Age'] = (pd.to_datetime('today') - df['Date of Birth']).dt.days // 365

# Define features and target variable
X = df.drop(columns=['Processing Status'])
y = df['Processing Status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing for numerical and categorical features
numeric_features = ['Age', 'Processing Time', 'Escalation Count', 'Issue Resolution Time']
categorical_features = ['Occupation Type', 'Submission Channel', 'Document Verification Status', 'ID Proof Type', 'Address Proof Type', 'Assigned Officer/Branch']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Create a pipeline with preprocessing and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# User Interface for input and prediction
def predict_status():
    try:
        age = int(entry_age.get())
        processing_time = int(entry_processing_time.get())
        escalation_count = int(entry_escalation_count.get())
        issue_resolution_time = int(entry_issue_resolution_time.get())
        occupation_type = entry_occupation_type.get()
        submission_channel = entry_submission_channel.get()
        document_verification_status = entry_document_verification_status.get()
        id_proof_type = entry_id_proof_type.get()
        address_proof_type = entry_address_proof_type.get()
        assigned_officer_branch = entry_assigned_officer_branch.get()

        input_data = pd.DataFrame({
            'Age': [age],
            'Processing Time': [processing_time],
            'Escalation Count': [escalation_count],
            'Issue Resolution Time': [issue_resolution_time],
            'Occupation Type': [occupation_type],
            'Submission Channel': [submission_channel],
            'Document Verification Status': [document_verification_status],
            'ID Proof Type': [id_proof_type],
            'Address Proof Type': [address_proof_type],
            'Assigned Officer/Branch': [assigned_officer_branch]
        })

        prediction = pipeline.predict(input_data)
        messagebox.showinfo("Prediction", f"The predicted processing status is: {prediction[0]}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create the main window
root = tk.Tk()
root.title("Pancard Application Tracker")

# Create and place labels and entries for user input
tk.Label(root, text="Age").grid(row=0, column=0)
entry_age = tk.Entry(root)
entry_age.grid(row=0, column=1)

tk.Label(root, text="Processing Time").grid(row=1, column=0)
entry_processing_time = tk.Entry(root)
entry_processing_time.grid(row=1, column=1)

tk.Label(root, text="Escalation Count").grid(row=2, column=0)
entry_escalation_count = tk.Entry(root)
entry_escalation_count.grid(row=2, column=1)

tk.Label(root, text="Issue Resolution Time").grid(row=3, column=0)
entry_issue_resolution_time = tk.Entry(root)
entry_issue_resolution_time.grid(row=3, column=1)

tk.Label(root, text="Occupation Type").grid(row=4, column=0)
entry_occupation_type = tk.Entry(root)
entry_occupation_type.grid(row=4, column=1)

tk.Label(root, text="Submission Channel").grid(row=5, column=0)
entry_submission_channel = tk.Entry(root)
entry_submission_channel.grid(row=5, column=1)

tk.Label(root, text="Document Verification Status").grid(row=6, column=0)
entry_document_verification_status = tk.Entry(root)
entry_document_verification_status.grid(row=6, column=1)

tk.Label(root, text="ID Proof Type").grid(row=7, column=0)
entry_id_proof_type = tk.Entry(root)
entry_id_proof_type.grid(row=7, column=1)

tk.Label(root, text="Address Proof Type").grid(row=8, column=0)
entry_address_proof_type = tk.Entry(root)
entry_address_proof_type.grid(row=8, column=1)

tk.Label(root, text="Assigned Officer/Branch").grid(row=9, column=0)
entry_assigned_officer_branch = tk.Entry(root)
entry_assigned_officer_branch.grid(row=9, column=1)

# Create and place the predict button
predict_button = tk.Button(root, text="Predict Status", command=predict_status)
predict_button.grid(row=10, columnspan=2)

# Run the main loop
root.mainloop()
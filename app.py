import pickle
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
import os 

EXCEL_FILE_PATH = 'patient_data.xlsx'  


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///patients.db'
db = SQLAlchemy(app)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(10), unique=True, nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    profession = db.Column(db.String(50), nullable=False)

@app.route("/add_patient", methods=['GET', 'POST'])
def add_patient():
    if request.method == 'POST':
        # Process the form data and add the patient to the database
        patient_id = request.form['patient_id']
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        profession = request.form['profession']

        # Check if the email already exists in the database
        existing_patient = Patient.query.filter_by(email=email).first()
        if existing_patient:
            return render_template('error.html', message="Email already exists. Please use a different email.")

        # Add the patient to the database
        new_patient = Patient(
            patient_id=patient_id,
            first_name=first_name,
            last_name=last_name,
            email=email,
            profession=profession
        )

        db.session.add(new_patient)
        db.session.commit()

        # Save patient data to an Excel sheet
        patient_data = {
            'Patient ID': [patient_id],
            'First Name': [first_name],
            'Last Name': [last_name],
            'Email': [email],
            'Profession': [profession]
        }

        df_patient = pd.DataFrame(patient_data)

        # Load existing data (if the file exists)
        if os.path.exists(EXCEL_FILE_PATH):
            existing_data = pd.read_excel(EXCEL_FILE_PATH)
            df_patient = pd.concat([existing_data, df_patient], ignore_index=True)

        # Save to Excel file
        df_patient.to_excel(EXCEL_FILE_PATH, index=False)

        # You can add a message or return a response here if needed

    return render_template('add_patient.html')



# Load the dataset
file_path = 'Disease_symptom_and_patient_profile_dataset.csv'
df = pd.read_csv(file_path)

# Identify categorical columns
categorical_columns = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Gender', 'Blood Pressure', 'Cholesterol Level']

# Use ordinal encoding for binary categorical columns
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df[categorical_columns] = ordinal_encoder.fit_transform(df[categorical_columns])

# Replace unknown values in the dataset
df.replace(-1, 'Unknown', inplace=True)

# Assuming 'Disease' is the new target variable (change this to your actual column name)
X = df.drop(['Disease'], axis=1)  # Features
y = df['Disease']  # New target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree model with the new target variable
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Save the model using pickle
with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(dt_model, file)

@app.route("/")
def index():
    patients = Patient.query.all()
    return render_template('index.html', patients=patients)



@app.route("/list_patients")
def list_patients():
    patients = Patient.query.all()
    return render_template('list_patients.html', patients=patients)





@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Gather input data from the form
        input_data = {
            'Fever': request.form['Fever'],
            'Cough': request.form['Cough'],
            'Fatigue': request.form['Fatigue'],
            'Difficulty Breathing': request.form['Difficulty_Breathing'],
            'Blood Pressure': request.form['Blood_Pressure'],
            'Gender': request.form['Gender'],
            'Cholesterol Level': request.form['Cholesterol_Level'],
            'Age': request.form['Age']
        }

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Label encode categorical columns
        for col in categorical_columns:
            le = LabelEncoder()
            input_df[col] = le.fit_transform(input_df[col])

        # Reorder columns to match the order during training
        input_df = input_df[X_train.columns]

        try:
            # Make prediction using the Decision Tree model
            prediction = dt_model.predict(input_df)

            # Calculate F1 score using cross-validation
            y_pred = cross_val_predict(dt_model, X, y, cv=5)  # You can adjust the cv parameter
            f1 = f1_score(y, y_pred, average='micro')

            # Display the prediction and F1 score on the result page
            return render_template('result.html', prediction=prediction[0], f1_score=f1)

        except NotFittedError as e:
            # Handle the case where the model is not fitted
            return render_template('error.html', message="Model not fitted. Please fit the model before making predictions.")
    
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)

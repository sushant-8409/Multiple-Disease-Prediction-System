import numpy as np
import pickle
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder="templates", static_folder="staticFiles")

working_dir = os.path.dirname(os.path.abspath(__file__))
diabetes_model = pickle.load(open(os.path.join(working_dir, "Models/diabetes_model.sav"), "rb"))
heart_disease_model = pickle.load(open(os.path.join(working_dir, "Models/heart_disease_model.sav"), "rb"))
parkinsons_model = pickle.load(open(os.path.join(working_dir, "Models/parkinsons_model.sav"), "rb"))

@app.route('/')
def home():
    return render_template('index.html')


# Route for rendering diabetes form page
@app.route('/diabetes')
def diabetes_page():
    # When someone clicks the 'Diabetes Prediction' option, render diabetes.html
    return render_template('diabetes.html')



@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():

    input_data = [float(request.form[field]) for field in ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]

    
    
    prediction = diabetes_model.predict([input_data])

    if prediction[0] == 1:
        result = 'The person is diabetic.'
    elif prediction[0] == 0 :
        result = 'The person is not diabetic.'
    
    return render_template('diabetes.html', diab_result=result)

# Route for rendering diabetes form page
@app.route('/heart')
def heart_page():
    # When someone clicks the 'Diabetes Prediction' option, render diabetes.html
    return render_template('heart.html')

@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    input_data = [float(request.form[field]) for field in ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]

    heart_result = None
    prediction = heart_disease_model.predict([input_data])

    if prediction[0] == 1:
        result = 'The person has heart disease.'
    else:
        result = 'The person does not have heart disease.'
    
    return render_template('heart.html', heart_result=result)

@app.route('/parkinsons')
def parkinsons_page():
    # When someone clicks the 'Diabetes Prediction' option, render diabetes.html
    return render_template('parkinsons.html')

@app.route('/predict_parkinsons', methods=['POST'])
def predict_parkinsons():
    input_data = [float(request.form[field]) for field in ['fo', 'fhi', 'flo', 'Jitter_percent', 'Jitter_Abs', 'RAP', 'PPQ', 'DDP', 'Shimmer', 'Shimmer_dB', 'APQ3', 'APQ5', 'APQ', 'DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']]
    prediction = parkinsons_model.predict([input_data])

    if prediction[0] == 1:
        result = "The person has Parkinson's disease."
    else:
        result = "The person does not have Parkinson's disease."
    
    return render_template('parkinsons.html', parkinsons_result=result)

if __name__ == "__main__":
    app.run(debug=True)

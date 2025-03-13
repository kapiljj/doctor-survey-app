from flask import Flask, request, render_template, send_file
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    time = float(request.form['time'])
    df = pd.read_csv('preprocessed_data.csv')
    df['Prediction'] = model.predict(df[['Active Hours', 'Count of Attempts']])
    results = df[df['Prediction'] == 1]
    results.to_csv('predicted_doctors.csv', index=False)
    return send_file('predicted_doctors.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

# Flask application
from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder='static', template_folder='templates')


# Load and train model once (so it’s ready when we use it)
data = pd.read_csv(r"C:\Users\cat\Desktop\AIML Project\Dataset\Crop_recommendation.csv")
X = data.drop('label', axis=1)
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    user_data = [[N, P, K, temperature, humidity, ph, rainfall]]
    prediction = model.predict(user_data)[0]

    return render_template('index.html', result=f" Best Crop to Grow: {prediction.upper()}")

if __name__ == '__main__':
    app.run(debug=True)

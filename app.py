from flask import Flask, render_template, request
from joblib import load
import numpy as np

app = Flask(__name__)

model = load("model.pkl")
scaler = load("scaler.pkl")


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict')
def predict():
    return render_template('index.html')


@app.route('/prediction', methods=['POST'])
def prediction():

    temp = float(request.form['temp'])
    humidity = float(request.form['humidity'])
    cloud = float(request.form['cloud'])
    annual = float(request.form['annual'])
    jan = float(request.form['jan'])
    march = float(request.form['march'])
    june = float(request.form['june'])
    octd = float(request.form['oct'])
    avgjune = float(request.form['avgjune'])
    sub = float(request.form['sub'])

    data = np.array([[
        temp,
        humidity,
        cloud,
        annual,
        jan,
        march,
        june,
        octd,
        avgjune,
        sub
    ]])

    data = scaler.transform(data)

    pred = model.predict(data)

    if pred[0] == 1:
        return render_template("chance.html")

    return render_template("no_chance.html")


if __name__ == '__main__':
    app.run(debug=True)
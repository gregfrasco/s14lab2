from flask import Flask, render_template, redirect
app = Flask(__name__)

import joblib

linear_regression = joblib.load('./notebooks/linear-regr.pkl')
decision_tree = joblib.load('./notebooks/decision-tree.pkl')
knn = joblib.load('./notebooks/knn.pkl')


@app.route('/')
def index():
    return redirect('/predict/0/4/2.5/3005/15/17903.0/1')

@app.route('/predict/<int:model>/<beds>/<baths>/<sqft>/<age>/<lotSize>/<int:garage>')
def predict(model, beds, baths, sqft, age, lotSize, garage):
    m = getModel(model)
    prediction = m.predict([[float(beds), float(baths), float(sqft), float(age), float(lotSize), garage]])[0][0].round(0)
    prediction = "${:,.0f}".format(prediction)
    return render_template('index.html', model=str(model), beds=str(beds), baths=str(baths), sqft=str(sqft), age=str(age), lotSize=str(lotSize), garage=str(garage), prediction=prediction)

def getModel(num):
    return [linear_regression,decision_tree,knn][num]

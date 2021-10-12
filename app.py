import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, render_template

import joblib
import json

app = Flask(__name__)

logreg = joblib.load('models/model_logreg.pkl')
knn = joblib.load('models/model_knn.pkl')
svc = joblib.load('models/model_svm.pkl')
decision = joblib.load('models/model_decision.pkl')
random_forest = joblib.load('models/model_randomforest.pkl')
gaussian = joblib.load('models/model_gaussiannb.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    d = None
    if request.method == 'POST':
        print('POST Received.')
        d = request.form.to_dict()
    else:
        print('GET Received.')
        d = request.args.to_dict()
    #d: Pclass, Sex, Age, Fare, Embarked, Title[Name], IsAlone[Parch, Sibsp],
    whole_table = pd.DataFrame([d])
|
|
|
    #['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'IsAlone', 'Age*Class']
    query = query.reindex(columns=model_columns, fill_value=0)

    if logreg:
        prediction = logreg.predict(query)
    if knn:
        prediction = knn.predict(query)
    if svc:
        prediction = svc.predict(query)
    if decision:
        prediction = decision.predict(query)
    if random_forest:
        prediction = random_forest.predict(query)
    if gaussian:
        prediction = gaussian.predict(query)

    if prediction[0]>prediction[1]:
            output = 'would have died.'
    elif prediction[0]<=prediction[1]:
            output = 'would have survived.'
    return render_template('index.html', prediction_text='You {}'.format(output))
if __name__ == '__main__':
    logreg = joblib.load('models/model_logreg.pkl')
    knn = joblib.load('models/model_knn.pkl')
    svc = joblib.load('models/model_svm.pkl')
    decision = joblib.load('models/model_decision.pkl')
    random_forest = joblib.load('models/model_randomforest.pkl')
    gaussian = joblib.load('models/model_gaussiannb.pkl')
    model_columns = joblib.load("model_columns.pkl")
    app.run(debug=True)



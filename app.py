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
    table = pd.DataFrame([d])
    '''
    Pclass   Sex   Age Fare   Embarked  Title   Name     IsAlone
0   first    male  21  322        C             dsdSD       1
    |
    |
    |
    '''
    #['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'IsAlone', 'Age*Class']
    #Age column

    if int(table['Age'])<=16:
        table['Age'] = 0
    elif int(table['Age']) > 16 and int(table['Age']) <= 32:
        table['Age'] = 1
    elif int(table['Age']) > 32 and int(table['Age']) <= 48:
        table['Age'] = 2
    elif int(table['Age']) > 48 and int(table['Age']) <= 64:
        table['Age'] = 3
    elif int(table['Age']) > 64:
        table['Age'] = 4

    if int(table['Fare']) <= 7.91:
        table['Fare'] = 0
    elif int(table['Fare']) > 7.91 and int(table['Fare']) <= 14.454:
        table['Fare'] = 1
    elif int(table['Fare']) > 14.454 and int(table['Fare']) <= 31:
        table['Fare'] = 2
    elif int(table['Fare']) > 31:
        table['Fare'] = 3
    table['Fare'] = table['Fare'].astype(int)

    #New column by feature engineering
    table['Age*Class'] = int(table['Age'].item()) * int(table['Pclass'].item())
    model = table['Model'].item()
    print(model)
    print(table)
    table = table.drop("Model", axis=1)
    query = table.reindex(columns=model_columns)
    print(query)
    if model=='logreg':
        prediction = logreg.predict(query)
    if model=='knn':
        prediction = knn.predict(query)
    if model=='svc':
        prediction = svc.predict(query)
    if model=='decision':
        prediction = decision.predict(query)
    if model=='random_forest':
        prediction = random_forest.predict(query)
    if model=='gaussian':
        prediction = gaussian.predict(query)
    print(prediction)
    if prediction==[0]:
            output = 'would have died.'
    elif prediction==[1]:
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

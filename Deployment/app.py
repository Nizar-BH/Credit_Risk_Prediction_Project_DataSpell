import pandas as pd
from pandas.core.algorithms import mode
import numpy as np
from flask import Flask, jsonify, request, render_template, make_response
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json

import logging
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
model = CatBoostClassifier()
model.load_model('FinalModelCatBoost')
col_to_norm = ['LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5',
                   'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 
                   'BILL_TOT', 'PAY_AMT_TOT', 'REMAINING_PAY_BALANCE']

def Normalization(X, scaler="minmax"):
    if scaler.upper() == "STANDARD":
        stand = StandardScaler()
        X_s = stand.fit_transform(X)
        return X_s
    else:
        minmax = MinMaxScaler()
        X_mm = minmax.fit_transform(X)
        return X_mm

# app
app = Flask(__name__)

# routes


@app.route('/predict', methods=['POST'])
def predict():
    # get data
    app.logger.warning("dqdqsdqssdsqdqsdsqqdds")

    app.logger.warning(request.form.values())

    data = request.form
    
    array = []
    array.append(int(data['LIMIT_BAL']))
    array.append(int(data['BILL_AMT1']))  # bill ammount
    array.append(int(data['BILL_AMT2']))
    array.append(int(data['BILL_AMT3']))
    array.append(int(data['BILL_AMT4']))
    array.append(int(data['BILL_AMT5']))
    array.append(int(data['BILL_AMT6']))
    array.append(int(data['PAY_AMT1'])) # pay ammount
    array.append(int(data['PAY_AMT2']))
    array.append(int(data['PAY_AMT3']))
    array.append(int(data['PAY_AMT4']))
    array.append(int(data['PAY_AMT5']))
    array.append(int(data['PAY_AMT6']))
    bill_tot = int(data['BILL_AMT1']) + int(data['BILL_AMT2']) + int(data['BILL_AMT3']) + int(data['BILL_AMT4']) + int(data['BILL_AMT5']) + int(data['BILL_AMT6'])
    array.append(bill_tot)  # BILL_TOT
    pay_amt_tot = int(data['PAY_AMT1']) + int(data['PAY_AMT2']) + int(data['PAY_AMT3']) + int(data['PAY_AMT4']) + int(data['PAY_AMT5']) + int(data['PAY_AMT6'])
    array.append(pay_amt_tot)  # PAY_AMT_TOT
    array.append(bill_tot - pay_amt_tot)  # REMAINING_PAY_BALANCE

    prediction = model.predict(array)
    app.logger.info(prediction)

    return render_template('index.html', prediction_text='default payment next month should be = {}'.format(prediction))


@app.route('/pred', methods=['GET'])
def predict2():

    # return data
    []
    return jsonify(model.feature_names_)


@app.route('/', methods=['GET'])
def Home():

    # return data
    []
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)




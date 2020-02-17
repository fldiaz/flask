from flask import Flask
from flask import render_template
from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from lightgbm import LGBMRegressor



app = Flask(__name__)

@app.route('/')
def index():
   return render_template('form.html')


# prediction function
def ValuePredictor(to_predict_list):
	to_predict = np.array(to_predict_list).reshape(1,24) #cantidad de variablesdel modelo
	loaded_model = pickle.load(open("model.pkl", "rb"))
	result = loaded_model.predict(to_predict)
	return result[0]



@app.route('/result', methods = ['POST'])
def result():
	if request.method == 'POST':
		to_predict_list = request.form.to_dict()
		to_predict_list = list(to_predict_list.values())
		#to_predict_list = list(map(int, to_predict_list))
		result = int(ValuePredictor(to_predict_list))
		return render_template("result.html", prediction = result)

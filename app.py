from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap 
import pandas as pd 
import numpy as np 
import joblib
import sys
sys.modules['sklearn.externals.joblib'] = joblib

# ML Packages
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	naivebayes_model = open("models/gnb2.pkl","rb")
	clf = joblib.load(naivebayes_model)

	if request.method == 'POST':
		namequery = request.form['namequery']
		data = [ord(namequery[-1].lower())-97]
		my_prediction = clf.predict(np.array(data).reshape(-1,1))
	return render_template('results.html',prediction = my_prediction[0],name = namequery.upper())


if __name__ == '__main__':
	app.run(debug=True)

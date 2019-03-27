import sys
import pandas as pd
import numpy as np
import pickle as pkl
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import flask
import datetime
from datetime import timedelta
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder
import Model_Testing as MT

app = Flask(__name__,template_folder='templates')

@app.route("/")
@app.route("/index")
def index():
	return flask.render_template('index.html')

MODEL = pkl.load(open('./data/XGB_TRAINED_MODEL.pkl','rb'))
DATASET = pd.read_csv("./data/UpdatedDataset.tsv", sep = "\t", index_col = False)

@app.route('/predict', methods=['GET','POST'])
def make_prediction():
	if request.method == 'POST':
		PRODUCT_ID = request.form['Product_ID']
		N = request.form['Number']
		user_list = []
		user_list = MT.get_top_users_for_product_id(DATASET, MODEL, PRODUCT_ID, N)
		return render_template('index.html', label = user_list)

@app.route('/graph', methods=['GET','POST'])
def show_graph():
	USER_ID = request.form['User_ID']
	USER_ID = int(USER_ID)
	plt.figure()
	data = DATASET[DATASET['User_ID'] == USER_ID]['Product_Category_1']
	count = sns.countplot(data)
	count = count.set(xlabel="Product Category")
	plt.savefig('./static/graphs/graphs.png')
	return render_template('index.html', name = "Top Categories for USER", url = './static/graphs/graphs.png')

if __name__ == '__main__':
	## start API
	app.run()
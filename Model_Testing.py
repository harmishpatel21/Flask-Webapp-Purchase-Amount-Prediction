## Import packages
import sys
import pandas as pd
import numpy as np
import pickle as pkl
import xgboost as xgb
import matplotlib.pyplot as mlp
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-u","--user",help="view top CATEGORIES for a USER_ID")
parser.add_option("-p","--product",help="view top USERS for Product_ID")
parser.add_option("-n","--number",help="set top N users")
(options,args) = parser.parse_args()

xgb_model = pkl.load(open('./data/XGB_TRAINED_MODEL.pkl','rb'))
DATASET = pd.read_csv("./data/UpdatedDataset.tsv", sep = "\t", index_col = False)

def get_top_users_for_product_id(DATASET, MODEL, PRODUCT_ID, N):
	PRODUCT_ID = int(PRODUCT_ID)
	N = int(N)	
	## Creating dictionary for product and user
	userDataDict = {}
	count = 0
	for i in set(DATASET['User_ID']):
		userDataDict[i] = [DATASET['Gender'][count],
			DATASET['Age'][count],
			DATASET['Occupation'][count],
			DATASET['City_Category'][count],
			DATASET['Stay_In_Current_City_Years'][count],
			DATASET['Marital_Status'][count]]
		count += 1

	productDataDict = {}
	count = 0
	for i in set(DATASET['Product_ID']):
		productDataDict[i] = [DATASET['Product_Category_1'][count],DATASET['Product_Category_2'][count],DATASET['Product_Category_3'][count]]
		count += 1

    ## Predictiing retult from product id
	tempArr = []
	for i in userDataDict.keys():
		testDF = pd.DataFrame([[i,PRODUCT_ID,userDataDict[i][0],
			userDataDict[i][1],userDataDict[i][2],
			userDataDict[i][3],userDataDict[i][4],
			userDataDict[i][5],productDataDict[PRODUCT_ID][0],
			productDataDict[PRODUCT_ID][1],productDataDict[PRODUCT_ID][2]]],
			columns=list(DATASET.columns[:-1]))
		tempArr.append((i,MODEL.predict(xgb.DMatrix(testDF))[0]))

	## Sorting the list based on purchase
	sortedArr = sorted(tempArr, key = lambda x: x[1], reverse = True)
	user_list = [] 
	for i in range(N):
		user_list.append(sortedArr[i][0])
	return user_list

def top_categories(DATASET,USER_ID):
	USER_ID = int(USER_ID)
	mlp.figure()
	data = DATASET[DATASET['User_ID'] == USER_ID]['Product_Category_1']
	sns.countplot(data)
	mlp.save



if __name__ == "__main__":

	if options.product and options.number:
		user_list = get_top_users_for_product_id(DATASET, xgb_model, options.product, options.number)
		print(user_list)

	elif options.user:
		data = top_categories(DATASET, options.user)
		mlp.figure()
		sns.countplot(data)
		mlp.show()

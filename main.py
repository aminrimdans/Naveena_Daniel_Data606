import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve, confusion_matrix
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
	st.title('FDA Recalls Project')

	st.image('FDA_recalls.png')

with dataset:
	st.header('Recalls Dataset')

	recalls_data = pd.read_excel('Recalls.xlsx')
	display_data = st.checkbox("See the first five recall data")
if display_data:
    st.write(recalls_data.head())

with features:
	recalls_data['Event Classification'] = recalls_data['Event Classification'].astype('category')
	lol = recalls_data['Event Classification'].astype('category')
	recalls_data['Event_indexed']=lol.cat.codes
	recalls_data['Reason for Recall'] = recalls_data['Reason for Recall'].apply(str.lower)
	recalls_data['alpha check'] = recalls_data['Reason for Recall'].str.isalpha()
	stopwords = stopwords.words('english')
	recalls_data['Reason for Recall'] = recalls_data['Reason for Recall'].apply(lambda x: ' '.join([w for w in x.split() if w not in (stopwords)]))

	recalls_data['Reason for Recall'] = recalls_data['Reason for Recall'].str.replace('\d+', '')
	recalls_data['Reason for Recall']=recalls_data['Reason for Recall'].astype('string')

	def vec_input(i):
		cvec = CountVectorizer()
		return cvec.fit_transform(i) #vectorization

		#X = cvec.fit_transform(recalls_data['Reason for Recall']) #vectorization

	original = vec_input(recalls_data['Reason for Recall'])

with model_training:
	y = recalls_data['Event_indexed']
	x = original
	x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.2, random_state=45)

	def plot_cfmatrix(prediction,y_test):
		st.subheader("Confusion Matrix")
		fig5 = plt.figure()
		conf_matrix = confusion_matrix(prediction , y_test)
		sns.set(font_scale=1.4)
		sns.heatmap(conf_matrix , annot=True , xticklabels=['Class I' , 'Class II', 'Class III'] , yticklabels=['Class I' , 'Class II', 'Class III'], cmap=plt.cm.Blues, annot_kws={'size':10}, linewidths=0.2, fmt=".2f")
		plt.ylabel("True")
		plt.xlabel("Predicted")
		st.pyplot(fig5)

	#add code to display x_test wrt to pred randomly.
	def display_pred(prediction,y_test):
		#dataframe for ytest
		original_df = recalls_data['Reason for Recall']

		#dataframe for pred 
		Predicted_df = pd.DataFrame(prediction)
		Predicted_df.index = pd.RangeIndex(start=47071, stop=47071+len(Predicted_df), step=1)
		#merge df based on index 
		new_df = pd.merge(Predicted_df, original_df, left_index=True, right_index=True)
		new_df.columns = ['Predicted Class', 'Reason for Recall']
		pd.set_option('display.max_colwidth', 2)

		output_df = new_df.sample()
		st.write("Reason for Recall:",output_df['Reason for Recall'],"\n Predicted Class:",output_df['Predicted Class'])

	mlmodel = st.sidebar.multiselect("Choose the model :", ('Logistic Regression', 'K-Nearest Neighbor', 'Random Forest Classifier'))

	def select_model(model_list):
		if 'Random Forest Classifier' in model_list:

			st.header('Selected ML model using recalls dataset')
			st.subheader('Random Forest Classifier')
			st.subheader('* **Parameter 1:** N_estimators')
			n_estimators = st.slider('Please choose the number of trees in the random forest classification model',min_value=10, max_value=120, value=20, step=10)
			st.subheader('* **Parameter 3:** Leaf Split')
			min_samples_leaf = st.selectbox('Please choose the minimum number of samples that can be stored in leaf node', options=[1,3,4,5],index=0)
			st.subheader('* **Parameter 4:** Sample Split')
			min_samples_split = st.selectbox('Please choose the minimum number of samples required to split the internal node', options=[2,6,10],index=0)
			rf = RandomForestClassifier(n_estimators=n_estimators,min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split)
			rf.fit(x_train,y_train)
			prediction = rf.predict(x_test)
			st.subheader('Accuracy score is:')
			st.write(accuracy_score(y_test, prediction))
			plot_cfmatrix(prediction,y_test)
			st.header('Random sample for Recall classification based on Random Forest Classifier model')
			display_pred(prediction,y_test)	

		if 'Logistic Regression' in model_list:
			st.subheader('Logistic Regression')
			lr = LogisticRegression(max_iter=30000)
			lr.fit(x_train,y_train)
			prediction = lr.predict(x_test)
			st.subheader('Accuracy score is:')
			st.write(accuracy_score(y_test, prediction))
			plot_cfmatrix(prediction,y_test)
			st.header('Random sample for Recall classification based on logistic regression model')
			display_pred(prediction,y_test)

		if 'K-Nearest Neighbor' in model_list:
			st.subheader('K-Nearest Neighbor')
			knn = KNeighborsClassifier()
			knn.fit(x_train,y_train)
			prediction = knn.predict(x_test)
			st.subheader('Accuracy score is:')
			st.write(accuracy_score(y_test, prediction))
			plot_cfmatrix(prediction,y_test)
			st.header('Random sample for Recall classification based on K-Nearest Neighbor model')
			display_pred(prediction,y_test)



select_model(mlmodel)








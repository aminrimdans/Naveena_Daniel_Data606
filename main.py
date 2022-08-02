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

	st.image('data/FDA_recalls.png')

with dataset:
	st.header('Recalls Dataset')

	recalls_data = pd.read_excel('data/Recalls.xlsx')
	display_data = st.checkbox("See the first five recall data")
if display_data:
    st.write(recalls_data.head())




with features:
	st.header('Parameters from Reason for Recall Text Column')

	st.subheader('Target variable with encoding')
	recalls_data['Event Classification'] = recalls_data['Event Classification'].astype('category')
	lol = recalls_data['Event Classification'].astype('category')
	recalls_data['Event_indexed']=lol.cat.codes
	st.write(recalls_data['Event_indexed'].head())


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
	st.header('ML model training using recalls dataset')

	st.subheader('* **Parameter 1:** N_estimators')
	n_estimators = st.slider('Please choose the number of trees in the random forest classification model',min_value=10, max_value=120, value=20, step=10)

	st.subheader('* **Parameter 3:** Leaf Split')
	min_samples_leaf = st.selectbox('Please choose the minimum number of samples that can be stored in leaf node', options=[1,3,4,5],index=0)

	st.subheader('* **Parameter 4:** Sample Split')
	min_samples_split = st.selectbox('Please choose the minimum number of samples required to split the internal node', options=[2,6,10],index=0)

	

	y = recalls_data['Event_indexed']
	x = original
	x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.2, random_state=45)
    

	rf = RandomForestClassifier(n_estimators=n_estimators,min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split)
	rf.fit(x_train,y_train)

	prediction = rf.predict(x_test)



	st.subheader('Mean Absolute Error of the model is:')
	st.write(mae(y_test, prediction))

	st.subheader('Mean Squared Error of the model is:')
	st.write(mse(y_test, prediction))

	st.subheader('R Mean Squared Error of the model is:')
	st.write(r2_score(y_test, prediction))

	st.subheader('Accuracy score is:')
	st.write(accuracy_score(y_test, prediction))

	st.header('Recall classification based on Random Forest Classifier model')
	st.subheader('get inputs from the user') 


	# Build the plot

#sns.set(font_scale=1.4)
#sns.heatmap(cm, annot=True, annot_kws={'size':10}, cmap=plt.cm.Greens, linewidths=0.2)

metrics = st.sidebar.multiselect("Choose evaluation metrics :", ('Confusion Matrix', 'ROC Curve'))

#st.subheader('Confusion Matrix')
#plot_confusion_matrix(rf, X_test, y_test)
#st.pyplot()

def plot_metrics(metrics_list):
	if 'Confusion Matrix' in metrics_list:
		st.subheader("Confusion Matrix")
		fig5 = plt.figure()
		conf_matrix = confusion_matrix(prediction , y_test)
		sns.set(font_scale=1.4)
		sns.heatmap(conf_matrix , annot=True , xticklabels=['Class I' , 'Class II', 'Class III'] , yticklabels=['Class I' , 'Class II', 'Class III'], cmap=plt.cm.Blues, annot_kws={'size':10}, linewidths=0.2, fmt=".2f")
		plt.ylabel("True")
		plt.xlabel("Predicted")
		st.pyplot(fig5)



	if 'ROC Curve' in metrics_list:
		st.subheader("ROC Curve")
		plot_roc_curve(rf, prediction, y_test)
		st.pyplot()


plot_metrics(metrics)







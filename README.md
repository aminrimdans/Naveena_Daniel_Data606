**# CLASSIFICATION OF FDA RECALLS - TEAM E  **

_**## By Daniel Rimdans and Moulya Naveena Choday **_

**###### Food and Drug Administration and Recalls **

The Food and Drug Administration (FDA) in United States is a government organization of the Department of Health and Human Services. The major responsibility of FDA is to secure and advance public health safety through the control and management of food handling. FDA primarily focuses on food, drugs and cosmetics but also involves other product types such as medical devices, tools, lasers, animal food and veterinary products. 

FDA recalls are the actions taken on the firm if there is any violation in the products. Recall can also be an intentional activity that happens on the grounds that manufacturers do their obligation to safeguard the general wellbeing and prosperity from items that present an act of injury or violations that might affect public health. Drug recall is the most effective method for safeguarding general society from a deficient or destructive item. 
Recalls are classified into three categories.  

Class I – Recalls that might cause severe injury or death. 
Class II – Recalls that might cause severe injury or temporary illness. 
Class III – Recalls that violate FDA rules but less likely to cause injury or illness. 

**###### Data **  

Data is collected from the FDA website that is publicly available and is updated yearly to provide the most recent records. The dataset file is 12.5 MB with a shape of (78184, 17). The dataset has high veracity as only 1 out of 78184 rows has a null entry. The column (Distribution Pattern) with the null entry is also not being utilized due to its inapplicability to our analysis. 

_**###### Data Description  **_

Recalling firm name - categorical 
Product Type - categorical 
Recalling Firm Country - categorical 
Reason for Recall - text 
Product Description - text 
Event Classification - categorical 

_**###### Questions/ Hypothesis **_

- What is being recalled more frequently and who is the manufacturing firm? 
- How many recalls does each firm have? 
- What is the severity of the reason for recall (Class I, II, III)? 
- Which product type has more recalls? 
- Which product type causes more severe (Class I & II) health impacts (food/cosmetics, devices, veterinary products, tobacco, and biologics)? 
- Which country has the highest recalled products? 
- Can we predict which firms are more likely to incur recalls? 

_**###### Exploratory Data Analysis**_

Imported required libraries and created a data frame by reading the excel file using pandas. Performed steps to clean the data. Checked for the null values and removed the single record from the dataset as there is only one null value in the whole dataset. Encoded the target variable to numerical values (1, 2, 3) as the type of target variable is category (Class I, Class II, Class III) respectively. Used One Hot Encoding technique to transform categorical columns product type and recalling firm country to numerical data types.  

_**Insights from data after performing EDA  **_

Exploratory Data Analysis is performed using SAS Viya and Python. 
Below are the visualizations using SAS Viya.  

1. Which product type has the highest recalls? 


 

From the figure, Pie chart displays the total records for the recalls and classification of the recalls where color red represents the most severe class and yellow represents the moderate severity and red represents less severity in the recalls. Displaying the recalls grouped by the product type, medical devices accounts for approximately 27,000 records is the highest among other product types such as food/cosmetics, Drugs, Veterinary, etc.  

2. Which product type causes more severe (Class I & II & III) health impacts? 

 

 

 

 

 

Above bar plot shows the classification of recalls for all the product types. The red color shows the recalls of class I which is severe. Food/Cosmetics has the highest frequency of recalls compared to the frequency of other product types.  

3. Which country has the highest recalled products? 

 

 

Among all the products that are recalled, United States has a greater number of recalled products which is around 97 percentage and remaining 3 percentage is other countries.	 

_**###### Natural Language Processing (NLP) – Vectorization **_

Reason for recall text column is cleaned by checking for stop words (frequently used words in English that are unimportant such as a, an, the) and removed them, replaced digits with alphabetic words, converted all the text into lower case and the text column is changed into numeric type by using count vectorization function from Scikit Learn library resulting in 22730 columns/features. 

 

 

 

 

 

 

 

 Based on the frequency of the words in the recalls text column, the top five key features are listed below. 

 

 

 

Tfidfvectorizer in scikit learn library which is like count vectorizer provides the importance of the words in the text column along with the frequency of the words. ‘Salmonella’ is the most important tokenized word in the Reason for Recalls column.  

 

 

 

 

_**###### Machine Learning Models  **_

Various Machine Learning models were executed to predict the class classification of reason for recall. Logistic Regression, Random Forest Classifier and K-Nearest Neighbor are the most accurate models among all other models.  

**_Logistic Regression: _**
Machine Learning algorithm which is same as linear regression but uses more complex cost function and uses predictive analysis algorithm based on the concept of probability.  

**_Random Forest Classifier: _**
Random Forest is a Supervised Machine Learning Algorithm that is utilized broadly in Classification and Regression issues. It considers majority votes for classification problems by creating the decision trees for the samples.   

**_K-Nearest Neighbor: _**
K-Nearest Neighbor is a supervised learning classification algorithm which uses closeness to make classifications about the group of an individual data point. It is also called KNN.  

_**###### Trail I –  **_

- Performed the One Hot Encoding technique on columns of Product Type and Recalling Firm Country.  
- Used those two columns' data to fit the three models and predict the results.  
- Accuracy of the models  

 

 

 

_**###### Trail II –  **_

- Performed Natural Language Processing technique on the text column (Reason for Recalls)  
- Used the text column after NLP vectorization as input to feed the models and predict the results. 
- Accuracy of the models  

 

 

Since the accuracy is more with trail II, we predicted the recalls classification using NLP techniques for the Machine Learning models. 

_**###### Model Deployment  **_

Created a simple web application using Streamlit which is an open-source framework for deploying Machine Learning models and projects with the help of python programming language. 

 

 

The first five records of the dataset are displayed if the user clicks the check box. 

 

Users can choose the preferred model from the drop-down among three models used in the project. 

 

 

 

 

Users can also select different parameters for the model based on the preferences. 

 

Metrics such as accuracy score and confusion matrix to display the true vs predicted values are shown for each model. 

 

 

A random sample from the input data set is taken and predicted class of recall classification is shown based on the model chosen. 

 

_**###### References:  **_

https://en.wikipedia.org/wiki/Food_and_Drug_Administration 
https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148 
https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/ 
https://streamlit.io/ 

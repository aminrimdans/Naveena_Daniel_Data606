# CLASSIFICATION OF FDA RECALLS - TEAM E

## By Daniel Rimdans and Moulya Naveena Choday

###### Link to the Presentation 
[Presentation](https://github.com/NaveenaChodayy/TEAM_E_Data606/blob/main/PPT_Presentation/Capstone%20Presentation.pdf)

###### Link to the video
[Video](https://www.youtube.com/watch?v=5kZn5iI_B6A)

###### Link to Streamlit Web Application
[Web Application](https://naveenachodayy-team-e-data606-streamlitproject-1kygzy.streamlitapp.com/)

### Food and Drug Administration and Recalls 

The Food and Drug Administration (FDA) in United States is a government organization of the Department of Health and Human Services. The major responsibility of FDA is to secure and advance public health safety through the control and management of food handling. FDA primarily focuses on food, drugs and cosmetics but also involves other product types such as medical devices, tools, lasers, animal food and veterinary products. 

FDA recalls are the actions taken on the firm if there is any violation in the products. Recall can also be an intentional activity that happens on the grounds that manufacturers do their obligation to safeguard the general wellbeing and prosperity from items that present an act of injury or violations that might affect public health. Drug recall is the most effective method for safeguarding general society from a deficient or destructive item. 
Recalls are classified into three categories.  

- Class I – Recalls that might cause severe injury or death. 
- Class II – Recalls that might cause severe injury or temporary illness. 
- Class III – Recalls that violate FDA rules but less likely to cause injury or illness. 

### Data 

Data is collected from the FDA website that is publicly available and is updated yearly to provide the most recent records. The dataset file is 12.5 MB with a shape of (78184, 17). The dataset has high veracity as only 1 out of 78184 rows has a null entry. The column (Distribution Pattern) with the null entry is also not being utilized due to its inapplicability to our analysis. 

##### Data Description 

- Recalling firm name - categorical 
- Product Type - categorical 
- Recalling Firm Country - categorical 
- Reason for Recall - text 
- Product Description - text 
- Event Classification - categorical 

### Questions/ Hypothesis 

- What is being recalled more frequently and who is the manufacturing firm? 
- How many recalls does each firm have? 
- What is the severity of the reason for recall (Class I, II, III)? 
- Which product type has more recalls? 
- Which product type causes more severe (Class I & II) health impacts (food/cosmetics, devices, veterinary products, tobacco, and biologics)? 
- Which country has the highest recalled products? 
- Can we predict which firms are more likely to incur recalls? 

### Exploratory Data Analysis

Imported required libraries and created a data frame by reading the excel file using pandas. Performed steps to clean the data. Checked for the null values and removed the single record from the dataset as there is only one null value in the whole dataset. Encoded the target variable to numerical values (1, 2, 3) as the type of target variable is category (Class I, Class II, Class III) respectively. Used One Hot Encoding technique to transform categorical columns product type and recalling firm country to numerical data types.  

##### Insights from data after performing EDA 

Exploratory Data Analysis is performed using SAS Viya and Python. 
Below are the visualizations using SAS Viya.  

#### Which product type has the highest recalls? 

![one_606](https://user-images.githubusercontent.com/106713975/185771556-25c40f05-ca77-4618-86cc-520c6da00d2d.png)

From the figure, Pie chart displays the total records for the recalls and classification of the recalls where color red represents the most severe class and yellow represents the moderate severity and red represents less severity in the recalls. Displaying the recalls grouped by the product type, medical devices accounts for approximately 27,000 records is the highest among other product types such as food/cosmetics, Drugs, Veterinary, etc.  

#### Which product type causes more severe (Class I & II & III) health impacts? 

<img width="655" alt="two_606" src="https://user-images.githubusercontent.com/106713975/185771568-1aa4ecb3-2f7c-49c1-8f8a-1dc75e1d09fd.png">

Above bar plot shows the classification of recalls for all the product types. The red color shows the recalls of class I which is severe. Food/Cosmetics has the highest frequency of recalls compared to the frequency of other product types.  

#### Which country has the highest recalled products? 

 <img width="552" alt="Screen Shot 2022-08-17 at 12 08 44 PM" src="https://user-images.githubusercontent.com/106713975/185771328-9a7945e0-541e-45a9-8a08-73f0afd463de.png">

Among all the products that are recalled, United States has a greater number of recalled products which is around 97 percentage and remaining 3 percentage is other countries.	 

### Natural Language Processing (NLP) – Vectorization

Reason for recall text column is cleaned by checking for stop words (frequently used words in English that are unimportant such as a, an, the) and removed them, replaced digits with alphabetic words, converted all the text into lower case and the text column is changed into numeric type by using count vectorization function from Scikit Learn library resulting in 22730 columns/features. 

 Based on the frequency of the words in the recalls text column, the top five key features are listed below. 

 ![Untitled 2](https://user-images.githubusercontent.com/106713975/185771340-24f6537b-fb3d-4c7f-8ac9-0cf47accc7e8.jpg)

Tfidfvectorizer in scikit learn library which is like count vectorizer provides the importance of the words in the text column along with the frequency of the words. ‘Salmonella’ is the most important tokenized word in the Reason for Recalls column.  

 ![Recall](https://user-images.githubusercontent.com/106713975/185771348-9e5a9091-0f44-4abc-b171-28434300f981.jpg)

### Machine Learning Models  

Various Machine Learning models were executed to predict the class classification of reason for recall. Logistic Regression, Random Forest Classifier and K-Nearest Neighbor are the most accurate models among all other models.  

##### Logistic Regression:
Machine Learning algorithm which is same as linear regression but uses more complex cost function and uses predictive analysis algorithm based on the concept of probability.  

##### Random Forest Classifier: 
Random Forest is a Supervised Machine Learning Algorithm that is utilized broadly in Classification and Regression issues. It considers majority votes for classification problems by creating the decision trees for the samples.   

##### K-Nearest Neighbor: 
K-Nearest Neighbor is a supervised learning classification algorithm which uses closeness to make classifications about the group of an individual data point. It is also called KNN.  

#### Trail I – 

- Performed the One Hot Encoding technique on columns of Product Type and Recalling Firm Country.  
- Used those two columns' data to fit the three models and predict the results.  
- Accuracy of the models  

<img width="588" alt="Screen Shot 2022-08-20 at 9 24 31 PM" src="https://user-images.githubusercontent.com/106713975/185771445-d5b874f6-647f-4248-b1ed-6004c15403a1.png">
 
#### Trail II –  

- Performed Natural Language Processing technique on the text column (Reason for Recalls)  
- Used the text column after NLP vectorization as input to feed the models and predict the results. 
- Accuracy of the models  

<img width="620" alt="Screen Shot 2022-08-20 at 9 25 10 PM" src="https://user-images.githubusercontent.com/106713975/185771453-a15ea63e-8dd2-4118-8a3e-dc7575eda07c.png">

Since the accuracy is more with trail II, we predicted the recalls classification using NLP techniques for the Machine Learning models. 

### Model Deployment  

Created a simple web application using Streamlit which is an open-source framework for deploying Machine Learning models and projects with the help of python programming language. 

The first five records of the dataset are displayed if the user clicks the check box.

<img width="1283" alt="Screen Shot 2022-08-20 at 7 27 48 PM" src="https://user-images.githubusercontent.com/106713975/185771476-a94b5072-7cb1-4db7-a2c1-923361b3f59a.png"> 

Users can choose the preferred model from the drop-down among three models used in the project. 

 <img width="336" alt="Screen Shot 2022-08-20 at 7 28 03 PM" src="https://user-images.githubusercontent.com/106713975/185771481-eb35e3a3-0ca5-4cc5-a010-df99bd01293c.png">

Users can also select different parameters for the model based on the preferences. 

<img width="1267" alt="Screen Shot 2022-08-20 at 7 32 13 PM" src="https://user-images.githubusercontent.com/106713975/185771499-54a72977-9ba5-4aeb-9c73-7d3bc50bfc15.png">

Metrics such as accuracy score and confusion matrix to display the true vs predicted values are shown for each model. 

<img width="1253" alt="Screen Shot 2022-08-20 at 7 33 30 PM" src="https://user-images.githubusercontent.com/106713975/185771504-dd224284-19e4-4e1d-bdb9-6d7dfeed3ef3.png">

A random sample from the input data set is taken and predicted class of recall classification is shown based on the model chosen. 

<img width="1213" alt="Screen Shot 2022-08-20 at 7 33 50 PM" src="https://user-images.githubusercontent.com/106713975/185771513-8100e5d7-50e4-4133-80f0-cec9463453b3.png">

### References:  

- https://en.wikipedia.org/wiki/Food_and_Drug_Administration 
- https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148 
- https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/ 
- https://streamlit.io/ 
 

   **Capstone Proposal Draft 
Daniel Rimdans
Moulya Naveena Choday**

**What is your issue of interest (provide sufficient background information)?**
The United States Food and Drug Administration (FDA) is a Federal agency of the Department of Health and Human Services which ensures safety and security of human and veterinary food supplies, drugs and medical devices, etc. 
Recalls are a method of protecting the public from defective or harmful products by removing or correcting the products that are in violation of laws administered by the FDA. 

**Why is this issue important to you and/or to others?**
The issue is pertinent due to the impact that commercially available products have on the health and wellbeing of the public. 
For all food/cosmetics, devices, veterinary products, tobacco, and biologics, the FDA oversees proper manufacturing and distribution of such products. 
Any discrepancy between manufacturing and distribution could lead to fatal incidents during public use.

**What questions do you have in mind and would like to answer?**
What is being recalled and who is the manufacturing firm?
How many recalls does each firm have?
What is the severity of the reason for recall (Class I, II, III)?
Which product type has more recalls and which product type causes more severe impacts (food/cosmetics, devices, veterinary products, tobacco, and biologics)?
Which country has the highest recalled products?
Can we predict which firms are more likely to incur recalls?
Which products are more probable to incur severe recalls (Class I & II)?

**Where do you get the data to analyze and help answer your questions (credibility of source, quality of data, size of data, attributes of data. etc.)?**
Our data is publicly available on the FDA website and is updated yearly to reflect the most recent records. 
The dataset file is 12.5 MB with a shape of (78184, 17). The dataset has high veracity as only 1 out of 78184 rows has a null entry. 
The column (Distribution Pattern) with the null entry is also not being utilized due to its inapplicability to our analysis.

**What will be your unit of analysis (for example, patient, organization, or country)?**
Our unit of analysis will be the severity of the impact of recalls which is measured in classes (Class I, II, III).
Class I - Product that could cause death or serious health problems.
Class II - Product that could cause temporary health issues and slight threat of a serious nature. 
Class III - Product that is less likely to cause adverse health issues but violates manufacturing laws of FDA. 

**Roughly how many units (observations) do you expect to analyze?**
We will be using all 78,184 observations in our dataset and approximately 6 input features affecting our target variable.

**What variables/measures do you plan to use in your analysis (variables should be tied to the questions in #3)?**
Recalling firm name
Counts of recall ‘Event ID’
Event classification (Class I, II, III)
Counts of recall ‘Event ID’ grouped by ‘Product Type’
Counts of recall ‘Event ID’ grouped by ‘Recalling Firm Country’
Probability of firm to incur recall (0-1)
Probability of product to incur Class I & II classification

**What kinds of techniques/models do you plan to use (for example, clustering, NLP, ARIMA, etc.)?**
Logistic regression 
Random forest classification 
K-Nearest Neighbors 
	
**How do you plan to develop/apply ML and how do you evaluate/compare the performance of the models?**
We plan to apply our ML algorithms using RandomizedSearchCV with cross-validation. Model performance evaluation will be done using accuracy score.

**What outcomes do you intend to achieve (better understanding of problems, tools to help solve problems, predictive analytics with practical applications, etc)?**
Our main intended outcome has to do with practical application of predictive analytics. The FDA regulates 272,719 products across its defined categories (animal drugs, animal food, biologics, human drugs, human food, medical devices, and tobacco). 
We want to create a practical model to assist in regulatory decisions through analyzing historical data of recall firms and recall products and predicting which new or existing products from specific manufacturers are more likely to cause severe health & safety impacts to the American public, and should thus be considered for recall evaluation.
We also plan to use Streamlit to build a simple web application to display the result of the target variable (Classes) using Python.

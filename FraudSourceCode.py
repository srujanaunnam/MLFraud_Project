#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pandas
#Pandas is a Python library, it has functions for analyzing, cleaning, exploring and manipulating data.
import numpy as np
#Numpy is a Python library, it also has functions for working in domain of linear algebra, fourier transformation and matrices.
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#Matplot is a visualization utility.
import seaborn as sns
# Seaborn is a library that uses Matplotlib underneath to plot graphs.
import datetime as dt
import math
import statsmodels.api as sm
sns.set_style('darkgrid')
plt.style.use('dark_background')
sns.set(color_codes=True)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


# In[5]:


#Import dataset
fraud= pandas.read_csv('Fraud.csv')


# In[6]:


# Returns first 10 rows of data
fraud.head(10)


# In[7]:


#Returns last 10 rows of data
fraud.tail(10)


# In[8]:


# number of rows and columns in dataset
fraud.shape


# In[9]:


# columns and data types in the Fraud dataframe
fraud.dtypes


# In[10]:


# columns headers easy to copy the header names in the syntax
fraud.columns


# In[11]:


#float format to 4 decimals.
pandas.set_option('display.float_format',lambda x:'%.4f' %x)


# # Exploratory data analysis:

# Exploratory data analysis (EDA) is used by data scientists to analyze and investigate data sets and summarize their descriptive statistics, often employing data visualization methods such as scatter plots, histograms, box plots etc..

# In[12]:


# Information about columns:
fraud.info()


# In[13]:


# Unique elements in the dataset.
fraud.nunique()


# In[93]:


# find missing values.
fraud.isnull().sum()


# In[96]:


#checking for duplicates
fraud.drop_duplicates(inplace=True)


# # Descriptive Statistics:

# In[97]:


# Description of variables rounidng to 4 decimals.
round(fraud[['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
       'oldbalanceDest', 'newbalanceDest']].describe().T,4)


# Observations:   
# Total number of rows rows, row count is 6362620   
# Transaction amount 0 for few records.   
# origin and destination balance is 0 for few records.   
# There are non null records.   
# 

# In[98]:


#0 transaction amount records.
fraud[fraud['amount']==0]


# In[99]:


#Is fraud records
fraud[fraud['isFraud']==1]


# In[100]:


# Is Flagged as Fraud records.
fraud[fraud['isFlaggedFraud']==1]


# # The entire amount from the origin account has been transfered to destination account and its a fraud transaction. Fraudster attempt to empty the account by transferring the whole amount.

# In[101]:


fraud[(fraud['amount']==fraud['oldbalanceOrg']) & (fraud['isFraud']==1)]


# # Variable Correlation statistics:

# In[14]:


fraud[['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
       'oldbalanceDest', 'newbalanceDest']].corr()


# # Origin old and new balance is highly correlated with the coefficients 0.9988. Destination old and new balance is highly correlated with the coefficients 0.9766.

# In[15]:


fraud['Origbalanceamt']=fraud['oldbalanceOrg']-fraud['newbalanceOrig']
fraud['Destbalanceamt']=fraud['oldbalanceDest']-fraud['newbalanceDest']
#fraud.drop(['oldbalanceDest','oldbalanceOrg','newbalanceOrig','newbalanceDest'],axis=1,inplace=True)


# In[304]:


# Select the independent variables of interest
ind_col= ['step','amount', 'isFraud','isFlaggedFraud','Origbalanceamt','Destbalanceamt']

# Create a correlation matrix
corr = fraud[ind_col].corr()

# Generate a correlation plot (heatmap)
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True,cmap='cool_r')

# Set plot title
plt.title('Correlation Plot')

# Display the plot
plt.show()


# # Visual Analysis:

#  Number of transactions by Type of transaction

# In[105]:


fraud['type'].value_counts()


# # Count Bar plot:

# In[106]:


ax= sns.countplot( x="type", order=fraud['type'].value_counts().index, color="aqua", saturation=0.4, data=fraud)
for p in ax.containers:
    ax.bar_label(p)
    ax.set_title(" Number of transactions by Type", loc="center", color="blue", size=20)
    pandas.set_option('display.float_format',lambda x:'%.4f' %x)
    


# # Pie Chart:

# In[107]:


plt.pie(x=fraud['type'].value_counts(),
        autopct='%.2f', labels=['CASH_OUT','PAYMENT','ASH_IN','TRANSFER','DEBIT'])
plt.show()


# # Histogram:

# In[108]:


sns.boxplot(x='amount', y='type', data=fraud)
plt.show()


# # Scatterplot:

# In[62]:


sns.scatterplot( x="amount", y='type', data=fraud,
                hue='type', size='amount')
 
# Placing Legend outside the Figure
plt.legend(bbox_to_anchor=(1, 1), loc=2)
 
plt.show()


# In[75]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=fraud, x='amount', y='step', hue='isFraud', alpha=0.5)
plt.xlabel('Step')
plt.ylabel('Amount')
plt.title('Scatter Plot of Step vs. Amount with Fraudulent Status (Hue)')
plt.grid(True)
plt.tight_layout()
plt.legend(title='Is Fraud')
plt.show()


# In[26]:


ax=sns.scatterplot(x='oldbalanceOrg', y='newbalanceOrig',data=fraud)


# In[110]:


target = 'isFraud'
fraud.groupby([target, 'type']).size().unstack(fill_value=0)


# In[111]:


fraud['amount'].describe()


# In[16]:


fraud['hour'] = fraud['step']-(24*(fraud['step']//24))
fraud.info()


# In[306]:


plt.figure(figsize=(14,6))
sns.histplot(data = fraud.sample(10000), x='hour', bins=23, kde=True)
plt.title('Distribution of the Transactions in 24 hours')
plt.xlabel('Hour of the day')
plt.xlim(0,23);

plt.figure(figsize=(14,6))
sns.histplot(data = fraud[fraud['isFraud'] == 1], x='hour', bins=23, kde=True)
plt.title('Distribution of the Fraud Transactions in 24 hours')
plt.xlabel('Hour of the day')
plt.xlim(0,23);

plt.figure(figsize=(14,6))
sns.histplot(data = fraud[fraud['isFraud'] == 0], x='hour', bins=23, kde=True, kde_kws={'cut':500})
plt.title('Distribution of the Legitimate Transactions in 24 hours')
plt.xlabel('Hour of the day')
plt.xlim(0,23);


# From above graphs we can see that Fraud trasactions happened throughout the day and more in after busniess hrs.

# # Transactions patterns by timings

# In[17]:


fraud.loc[fraud['hour'] <= 7, 'time'] = 'NIGHT'
fraud.loc[((fraud['hour'] > 7) & (fraud['hour'] <=12)), 'time'] = 'MORNING'
fraud.loc[((fraud['hour'] > 12) & (fraud['hour'] <=18)), 'time'] = 'DAY'
fraud.loc[((fraud['hour'] > 18) & (fraud['hour'] <=23)), 'time'] = 'EVENING'
fraud.time.unique()


# In[115]:


fraud.time = pandas.Categorical(fraud.time, ['NIGHT', 'MORNING', 'DAY', 'EVENING'])


# In[116]:


plt.figure(figsize=(14,6))
sns.histplot(data = fraud.sample(10000), x='time', bins=23, kde=False)
plt.title('Distribution of the Transactions in 24 hours')
plt.xlabel('Time of the day');

plt.figure(figsize=(14,6))
sns.histplot(data = fraud[fraud['isFraud'] == 1], x='time', bins=23, kde=False)
plt.title('Distribution of the Fraud Transactions in 24 hours')
plt.xlabel('Time of the day');


# In[18]:


fraud_fraudulent = fraud[fraud.isFraud == 1]
fraud_fraudulent.head()


# In[23]:


fraud_legitimate =fraud[fraud['isFraud'] == 0]
percentage_of_transfered_full_amount_fraud = round(100*fraud_fraudulent.newbalanceOrig[fraud_fraudulent.newbalanceOrig == 0].count()/fraud_fraudulent.newbalanceOrig.count(),2)
percentage_of_transfered_full_amount_fulldataset = round(100*fraud.newbalanceOrig[fraud.isFraud == 0][fraud.newbalanceOrig == 0].count()/fraud.newbalanceOrig.count(),2)
plt.bar(x=[1,2], height =[percentage_of_transfered_full_amount_fraud, percentage_of_transfered_full_amount_fulldataset],tick_label=['transfered_full_amount_fraud','transfered_full_amount_dataset'])
print('\n {}% of fraud transactions transfered all the money from the account.\n Percentage of the Legitimate transactions that transferred everything from the account of Origin is {}%'.format(percentage_of_sending_all_assets_fraud,percentage_of_sending_all_assets_fulldataset))


# In[24]:


legitimate = len(fraud[fraud.isFraud == 0])
fraudulent = len(fraud[fraud.isFraud == 1])
legit_percent = (legitimate / (fraudulent + legitimate)) * 100
fraud_percent = (fraudulent/ (fraudulent + legitimate)) * 100

print("Number of Legitimate transactions: ", legitimate)
print("Number of Fraud transactions: ", fraudulent)
print("Percentage of Legit transactions: {:.4f} %".format(legit_percent))
print("Percentage of Fraud transactions: {:.4f} %".format(fraud_percent))


# # Outliers:

# In[109]:


fig=plt.figure(figsize=(15,10))
sns.boxplot(fraud)
plt.show()


# In[27]:


# Select only numeric columns
numeric_columns = fraud.select_dtypes(include=['int64','float64'])

# Calculate quantiles for numeric columns
Q1 = numeric_columns.quantile(0.25)
Q3 = numeric_columns.quantile(0.75)
IQR = Q3 - Q1

# The rest of your code to detect outliers should remain the same
def detect_outliers(column):
    lower_bound = Q1[column] - 1.5 * IQR[column]
    upper_bound = Q3[column] + 1.5 * IQR[column]
    outliers = fraud[(fraud[column] < lower_bound) | (fraud[column] > upper_bound)]
    if outliers.empty:
        return None, 0.0
    else:
        percentage = (len(outliers) / len(fraud)) * 100
        return column, percentage

columns_with_outliers = [detect_outliers(column) for column in numeric_columns.columns]
columns_with_outliers = [(column, percentage) for column, percentage in columns_with_outliers if column is not None]

for column, percentage in columns_with_outliers:
    print(f"Column: {column}, Percentage of outliers: {percentage:.2f}%")


# In[26]:


def remove_outlier(col):
    sorted(col)
    Q1,Q3=col.quantile([0.25,0.75])
    IQR=Q3-Q1
    lower_range=Q1-1.5*IQR
    upper_range=Q3+1.5*IQR
    return lower_range,upper_range


# # Feature selection :-
# Selecting useful features and removing features which are not playing significant role in model training.

# In[126]:


def correlation(dataset,threshold):

    column_corr=set()
    
    # storing correlation matrix
    corr_matrix=fraud.corr()
    
    for i in range (len(corr_matrix.columns)):
        for j in range (i):
            
            # comparing corr. values with threshold
            if corr_matrix.iloc[i,j]>threshold:
                
                # if true, then fateching column name
                colname=corr_matrix.columns[i]
                
                # adding column name to column_corr variable.
                column_corr.add(colname)
    return column_corr


# In[127]:


correlated_features=correlation(fraud,0.9)


# In[128]:


len(set(correlated_features))


# In[129]:


correlated_features


# In[130]:


fraud=fraud.drop(correlated_features,axis=1)


# In[131]:


fraud.head()


# # Checking variations of data in features :-
# if variation is zero (0) , means that column ( feature ) is not
# having any correlation with dependent features. and we need to remove that column.

# In[132]:


from sklearn.feature_selection import VarianceThreshold


# threshold=0  means feature having only 1 value ( no variation)   
# threshold=1 means feature having 2 different values ( small variation)
# 

# In[133]:


Var_Thresh=VarianceThreshold(threshold=0)
Var_Thresh.fit(fraud[['step','amount','oldbalanceOrg','oldbalanceDest','isFlaggedFraud']])
Var_Thresh.get_support()


# In[134]:


# when threshold=1

Var_Thresh=VarianceThreshold(threshold=1)
Var_Thresh.fit(fraud[['step','amount','oldbalanceOrg','oldbalanceDest','isFlaggedFraud']])
Var_Thresh.get_support()


# NOTICED:- 'isFlaggedFraud' feature having Low_variation in data ,
# but we can't remove that column, because it contain these two values (0,1)
# which helps to identify whether our transc. is fraud or not
# and ,
# 
# it also shows good correlationship with dependent feature ( target feature )
# Thus removing this feature is not good option.

# In[135]:


fraud.dtypes


# In[137]:


print(" type having these unique values :- \n",fraud["type"].unique())
print("\n\n nameOrig having these unique values :- \n",fraud["nameOrig"].unique())
print("\n\n nameDest having these unique values :- \n",fraud["nameDest"].unique())


# In[161]:


fraud.head()


# In[162]:


fraud=fraud.drop({'nameOrig','nameDest','hour','time','Origbalanceamt','Destbalanceamt'},axis=1)


# In[165]:


fraud=fraud.drop({'newbalanceOrig','newbalanceDest'},axis=1)


# In[166]:


fraud.head()


# # Feature Encoding :

# In[167]:


fraud['type']=fraud['type'].map({'CASH_OUT':5, 'PAYMENT':4,'CASH_IN':3,'TRANSFER':2,'DEBIT':1})


# In[168]:


fraud.head()


# # Handling Un-Balanced Data:

# In[169]:


fraud["isFraud"].value_counts()


# In[171]:


legitimate_transaction=fraud[fraud['isFraud']==0]
fraud_transaction=fraud[fraud['isFraud']==1]


# In[172]:


legitimate_transaction.head()


# In[173]:


fraud_transaction.head()


# In[174]:


print(legitimate_transaction.shape)
print(fraud_transaction.shape)


# In[175]:


# taking random 8213 records from normal_transaction

legitimate_transaction=legitimate_transaction.sample(n=8213)


# Now, we have 50-50% fraud and normal transaction data.   
# next step is to concatenating them
# 

# In[176]:


print(legitimate_transaction.shape)
print(fraud_transaction.shape)


# In[177]:


legitimate_transaction['amount'].describe()


# In[178]:


fraud_transaction['amount'].describe()


# In[180]:


fraud_balanced=pandas.concat([legitimate_transaction,fraud_transaction], axis=0)


# In[183]:


fraud_balanced.head()


# In[184]:


fraud_balanced.shape


# # Train-Test Split

# In[185]:


# independent features
X=fraud_balanced.drop("isFraud",axis=1)

# dependent feature
y=fraud_balanced["isFraud"]


# In[186]:


X.shape


# In[187]:


y.shape


# In[188]:


from sklearn.model_selection import train_test_split


# In[254]:


x_train,x_test,y_train,y_test=train_test_split(X , y , test_size=0.2, stratify=y , random_state=0)


# In[255]:


print("x-train :- ", x_train.shape)
print("x-test :-  ",  x_test.shape)
print("y-train :- ", y_train.shape)
print("y-test :-  ",  y_test.shape)


# In[256]:


y_test.value_counts()


# In[257]:


y_train.value_counts()


# # Feature Scaling

# In[258]:


from sklearn.preprocessing import StandardScaler


# In[259]:


scaler=StandardScaler()


# In[260]:


scaler.fit(x_train)


# In[261]:


x_train_scaler=scaler.transform(x_train)


# In[262]:


x_test_scaler=scaler.transform(x_test)


# In[263]:


x_train_scaler


# In[264]:


x_test_scaler


# # Training and Evaluating model

# # Logistic Regression

# In[265]:


from sklearn.linear_model import LogisticRegression


# In[266]:


log_model=LogisticRegression()


# In[267]:


log_model.fit(x_train_scaler,y_train)


# In[268]:


y_pred=log_model.predict(x_test_scaler)


# In[269]:


from sklearn.metrics import accuracy_score


# In[270]:


print("Accuracy of Logistic Regression")
print(accuracy_score(y_test.values,y_pred)*100)


# # Random Forest Classifier

# In[271]:


from sklearn.ensemble import RandomForestClassifier


# In[272]:


rand_model=RandomForestClassifier()


# In[273]:


rand_model.fit(x_train_scaler,y_train)


# In[274]:


y_pred=rand_model.predict(x_test_scaler)


# In[275]:


print("Accuracy of Random Forest Classifier")
print(accuracy_score(y_test.values,y_pred)*100)


# In[279]:


#Evaluating the classifier
#printing every score of the classifier
#scoring in any thing
from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef
from sklearn.metrics import confusion_matrix
n_outliers = len(fraud)
n_errors = (y_pred != y_test).sum()
print("The model used is Random Forest classifier")
acc= accuracy_score(y_test,y_pred)
print("The accuracy is  {}".format(acc))
prec= precision_score(y_test,y_pred)
print("The precision is {}".format(prec))
rec= recall_score(y_test,y_pred)
print("The recall is {}".format(rec))
f1= f1_score(y_test,y_pred)
print("The F1-Score is {}".format(f1))
MCC=matthews_corrcoef(y_test,y_pred)
print("The Matthews correlation coefficient is {}".format(MCC))
#printing the confusion matrix
LABELS = ['Legitimate', 'Fraudulent']
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

# Run classification metrics
plt.figure(figsize=(9, 7))
print('{}: {}'.format("Random Forest", n_errors))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# # Support Vector Machine ( SVM )

# In[211]:


from sklearn.svm import SVC


# In[212]:


svm_model=SVC()


# In[213]:


svm_model.fit(x_train_scaler,y_train)


# In[214]:


y_pred=svm_model.predict(x_test_scaler)


# In[215]:


print("Accuracy of Support Vector Machine (SVM)")
print(accuracy_score(y_test.values,y_pred)*100)


# # BernoulliNB

# In[216]:


from sklearn.naive_bayes import BernoulliNB


# In[217]:


bnb_model=BernoulliNB()


# In[218]:


bnb_model.fit(x_train_scaler,y_train)


# In[219]:


y_pred=bnb_model.predict(x_test_scaler)


# In[220]:


print("Accuracy of BernoulliNB ")
print(accuracy_score(y_test.values,y_pred)*100)


# # GaussianNB

# In[221]:


from sklearn.naive_bayes import GaussianNB


# In[222]:


gnb_model=GaussianNB()


# In[223]:


gnb_model.fit(x_train_scaler,y_train)


# In[224]:


y_pred=gnb_model.predict(x_test_scaler)


# In[225]:


print("Accuracy of GaussianNB ")
print(accuracy_score(y_test.values,y_pred)*100)


# # Decision Tree

# In[227]:


from sklearn.tree import DecisionTreeClassifier


# In[228]:


DT_model=DecisionTreeClassifier(random_state = 2)


# In[229]:


DT_model.fit(x_train_scaler,y_train)


# In[230]:


y_pred=DT_model.predict(x_test_scaler)


# In[231]:


print("Accuracy of Decision Tree ")
print(accuracy_score(y_test.values,y_pred)*100)


# In[249]:


from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus


# In[251]:


dot_data = StringIO()
export_graphviz(DT_model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('fraud.png')
Image(graph.create_png())


# In[252]:


from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(DT_model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('fraud1.png')
Image(graph.create_png())


# # KNN Modeling

# In[232]:


from sklearn.neighbors import KNeighborsClassifier# Create a KNN classifier 


# In[233]:


knn = KNeighborsClassifier(n_neighbors=3)


# In[234]:


knn.fit(x_train_scaler,y_train)


# In[235]:


y_pred=knn.predict(x_test_scaler)


# In[236]:


print("Accuracy of KNN Model ")
print(accuracy_score(y_test.values,y_pred)*100)


# In[289]:


import pickle


# In[290]:


pickle.dump(rand_model,open('model.sav','wb'))


# In[291]:


#### Saving the StandadrdScaler object 'scaler'

pickle.dump(scaler,open('scaler.sav','wb'))


# In[292]:


rand_model=pickle.load(open('model.sav','rb'))


# In[293]:


# loading the scaler file for scaling input array
new_scaler=pickle.load(open('scaler.sav','rb'))


# In[294]:


fraud_balanced.head()


# In[295]:


input_array=np.array([[228,5,117563.1100,0.0000,208908.4100,0]])
input_array


# In[296]:


input_array_scale=new_scaler.transform(input_array)
input_array_scale


# In[297]:


pred=rand_model.predict(input_array_scale)
pred


# In[ ]:





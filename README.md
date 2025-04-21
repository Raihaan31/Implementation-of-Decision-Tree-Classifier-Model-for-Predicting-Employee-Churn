# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: R Raihaan Ahmed
RegisterNumber:  212224040260
*/
```
```py
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,206,6,0,1,2]])
```
## Output:
## DATA HEAD:
![image](https://github.com/user-attachments/assets/498bf2ba-94d4-4b64-b5e7-ed86c025c2a9)


## DATASET INFO:
![image](https://github.com/user-attachments/assets/c8d3481a-a95b-4498-9eae-53f9e56742db)


## NULL DATASET:
![image](https://github.com/user-attachments/assets/10872743-9adc-4165-bb4d-758ed72ee08f)


## VALUES COUNT IN THE LEFT COLUMN
![image](https://github.com/user-attachments/assets/4e636576-6f86-40b4-98a3-16dc69911f89)


## DATASET TRANSFORMED HEAD:
![image](https://github.com/user-attachments/assets/f3d2b781-188b-4880-932c-6fb0a477999a)


## X.HEAD:
![image](https://github.com/user-attachments/assets/090d23f2-4a89-457b-8bfc-6efabdc7e063)


## ACCURACY:
![image](https://github.com/user-attachments/assets/1f2344dc-8943-45a8-94b5-e7f1c1492201)


## DATA PREDICTION:
![image](https://github.com/user-attachments/assets/00ad8927-736c-42bc-a1aa-eb64ee07d5ea)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

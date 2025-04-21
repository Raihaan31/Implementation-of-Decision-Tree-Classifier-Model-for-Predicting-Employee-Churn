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
df = pd.read_csv("/content/Employee.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df['left'].value_counts())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['salary'] = le.fit_transform(df['salary'])

x = df[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]

print(x)
y = df['left']
print(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 30)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)
print("Y Predicted : \n\n",y_pred)

from sklearn import metrics

accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"\nAccuracy : {accuracy * 100:.2f}%")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
![image](https://github.com/user-attachments/assets/ec39f641-83d0-408b-8d2e-12a9c9970497)
![image](https://github.com/user-attachments/assets/8ff950f1-4171-4b0e-9579-fad56dc891c7)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

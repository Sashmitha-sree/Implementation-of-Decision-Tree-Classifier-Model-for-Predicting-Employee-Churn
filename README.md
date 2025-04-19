# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import libraries and load the dataset.
2.Handle null values and encode categorical columns.
3.Split data into training and testing sets.
4.Train a DecisionTreeClassifier using entropy.
5.Predict and evaluate the model using accuracy and metrics.
```
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SASHMITHA SREE K V 
RegisterNumber:  
*/
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# Load dataset
data = pd.read_csv("/content/Employee (1).csv")
data["salary"] = LabelEncoder().fit_transform(data["salary"])

# Features and target
x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", 
          "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
y = data["left"]

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Train Decision Tree
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train, y_train)

# Predict and evaluate
y_pred = dt.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Predict new sample
sample = dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])
print("Sample Prediction:", sample)
```

## Output:
![decision tree classifier model](sam.png)
![image](https://github.com/user-attachments/assets/2c31263a-e0b5-4b0c-9c7b-68779de650d5)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

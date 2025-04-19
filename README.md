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
RegisterNumber:  212224230255/24900551
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
## EXPLORE THE DATASET
![Screenshot 2025-04-16 223721](https://github.com/user-attachments/assets/18e99d63-425c-4443-9c84-6d4729d93ca5)
## FEATURES AND TARGET
![Screenshot 2025-04-16 223813](https://github.com/user-attachments/assets/5bdf41a5-1236-4ec7-abd3-c042b419fd46)
## DECISION TREE MODEL
![Screenshot 2025-04-16 223857](https://github.com/user-attachments/assets/1685ae0b-dd1d-41b1-bbcd-24d9630076dd)
## ACCURACY  
![Screenshot 2025-04-16 224033](https://github.com/user-attachments/assets/f718f84c-1561-4636-be38-2c5b52c529e2)
## PREDICTIONS AND EVALUATE
![Screenshot 2025-04-16 224038](https://github.com/user-attachments/assets/cabb1ba9-0d3d-4a7c-b9ba-1afba94cd779)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
`` 
data= {
    "Hours_Studied":[1,2,3,4,5,6,7,8,9,10],
    "Marks_Scored":[35,40,50,55,60,65,70,80,85,95]
}
df = pd.DataFrame(data)
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
print("Dataset:\n",df.head())
df
x= df[["Hours_Studied"]]
y= df["Marks_Scored"]
x_train, x_test, y_train, y_test=train_test_split(
    x,y, test_size=0.2,random_state=42
)
model= LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("\nModel Prameters:")
print("Intercept(b0):",model.intercept_)
print("Slope (b1):,",model.coef_[0])
print("\nEvaluation Metrics:")
print("Mean Squared Error:",mean_squared_error(y_test,y_pred))
print("R^2 Score:",r2_score(y_test,y_pred))
plt.figure(figsize=(8,6))
plt.scatter(x, y,color='blue', label="Actual Data")
plt.plot(x,model.predict(x),color='red', linewidth=2,label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Simple Linear Regression: Predicting Marks")
plt.legend()
plt.grid(True)
plt.show()

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 
RegisterNumber:  
*/
```

## Output:
<img width="869" height="656" alt="Screenshot 2025-11-13 091921" src="https://github.com/user-attachments/assets/b486a632-5652-4c75-9b0e-81782f835cfc" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

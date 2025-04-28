# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Start

Step 2: Import the Required Libraries
Import numpy as np for numerical operations.

Step 3: Load or Define the Dataset

Step 4: Preprocess the Data
Reshape X and y to ensure they are in proper matrix form.

Step 5: Initialize Model Parameters
Initialize the parameter vector theta with zeros (or small random numbers).
Set hyperparameters:
Learning Rate (α) — controls how big each update step is.
Number of Iterations — how many times to update theta.

Step 6: Define the Sigmoid Function

Step 7: Define the Cost Function (Log-Loss)

Step 8: Implement the Gradient Descent Algorithm

Step 9: Train the Model
Call the gradient descent function.
Let it optimize theta to minimize the cost function.
After completing all iterations, you get the optimized model parameters.

Step 10: Make Predictions

Step 11: Evaluate the Model
Compare the predicted labels with actual labels.

Step 12: Visualize Cost Function Convergence (Optional)

Step 13: End


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Manisha selvakumari.S.S.
RegisterNumber: 212223220055
*/
import pandas as pd
import numpy as np

d=pd.read_csv('Placement_Data.csv')
d

d=d.drop('sl_no',axis=1)
d=d.drop('salary',axis=1)
print("Name: Manisha selvakumari.S.S.")
print("Reg No: 212223220055")

d['gender']=d['gender'].astype('category')
d['ssc_b']=d['ssc_b'].astype('category')
d['hsc_b']=d['hsc_b'].astype('category')
d['hsc_s']=d['hsc_s'].astype('category')
d['degree_t']=d['degree_t'].astype('category')
d['workex']=d['workex'].astype('category')
d['specialisation']=d['specialisation'].astype('category')
d['status']=d['status'].astype('category')
d.dtypes

d['gender']=d['gender'].cat.codes
d['ssc_b']=d['ssc_b'].cat.codes
d['hsc_b']=d['hsc_b'].cat.codes
d['hsc_s']=d['hsc_s'].cat.codes
d['degree_t']=d['degree_t'].cat.codes
d['workex']=d['workex'].cat.codes
d['specialisation']=d['specialisation'].cat.codes
d['status']=d['status'].cat.codes
d

x=d.iloc[:, :-1].values
y=d.iloc[:, -1].values

y

theta=np.random.randn(x.shape[1])
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,x,y):
    h=sigmoid(x.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*r)

theta=np.random.randn(x.shape[1])
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,x,y):
    h=sigmoid(x.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,x,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(x.dot(theta))
        gradient=x.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta

theta=gradient_descent(theta, x, y, alpha=0.01, num_iterations=1000)

def predict(theta,x):
    h=sigmoid(x.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred
y_pred=predict(theta,x)
accuracy=np.mean(y_pred.flatten()==y)
print(("Accuracy",accuracy))

 print(y_pred)

print(y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

## Output:
![Screenshot 2025-04-28 225341](https://github.com/user-attachments/assets/9cf1564c-18a9-415a-9cbc-94f4ca31b132)

![Screenshot 2025-04-28 225358](https://github.com/user-attachments/assets/d60b431a-fee7-45ca-8435-047c8a451a54)

![Screenshot 2025-04-28 225415](https://github.com/user-attachments/assets/a0e7683e-ac70-4433-aaa4-e285a1fbda00)

![Screenshot (255)](https://github.com/user-attachments/assets/5771256a-befd-4bc5-8a22-b98a04bc2c7e)

![Screenshot (256)](https://github.com/user-attachments/assets/160fa22e-63b2-48a7-af7d-86e684a94cf4)

![Screenshot 2025-04-28 230610](https://github.com/user-attachments/assets/d4073580-c112-4535-ac13-4b6484a756ba)

![Screenshot 2025-04-28 230625](https://github.com/user-attachments/assets/65e017f5-d7c9-4898-a1e7-0c1a5fbf9221)

![Screenshot 2025-04-28 231448](https://github.com/user-attachments/assets/4ba28adf-579d-4345-a7f6-ad4011663c08)

![Screenshot (258)](https://github.com/user-attachments/assets/bcd653a8-1622-404f-b90a-77b3a5a362bb)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("./Data/Fish.csv")


# Data processing
#print(df.isna().sum()) # no NULL value

# Encode Categorical Data
df["Species"] = df["Species"].astype("category") # Type casting Species column to category datatype
df["Species"] = df["Species"].cat.codes # Encode Species column

# Split features and Label
X = df.drop(columns = "Weight")
Y = df["Weight"]

# Split data into training and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) # 20% for test

# Create model
Lr = LinearRegression()

# Train model
Lr.fit(x_train, y_train)

# Predict
#print(Lr.predict([[1,2,3,4,5,6]]))

# Test predict accuracy
print(r2_score(y_test,Lr.predict(x_test)))

# Scatter plot result comparison
y_test_predict = Lr.predict(x_test)
plt.scatter(y_test,y_test_predict)
plt.xlabel("Actual Weights")
plt.ylabel("Predicted Weights")
plt.show()


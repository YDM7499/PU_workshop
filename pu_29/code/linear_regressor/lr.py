import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import pandas as pd
 
# Load CSV and columns
df = pd.read_csv("./Housing.csv")
 
Y = df['price']
X = df['lotsize']

"""
X = X.reshape((len(X),1))
bedrooms = df['bedrooms']
bedrooms = bedrooms.reshape((len(bedrooms),1))
X = np.append(X, bedrooms, axis = 1)
print(X.shape, Y.shape)

"""

X = X.reshape((len(X),1))
Y=Y.reshape(len(Y),1)

# Split the data into training/testing sets
X_train = X[:-250]
X_test = X[-250:]
 
# Split the targets into training/testing sets
Y_train = Y[:-250]
Y_test = Y[-250:]
 
# Plot outputs
plt.scatter(X_test, Y_test,  color='black')
plt.title('Test Data')
plt.xlabel('Size')
plt.ylabel('Price')
plt.xticks(())
plt.yticks(())

"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y[:,0])
"""
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, Y_train)

"""
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
"""
# Plot outputs
plt.plot(X_test, regr.predict(X_test), color='red',linewidth=3)
plt.show()
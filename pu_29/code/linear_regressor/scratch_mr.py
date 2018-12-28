import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('student.csv')
print(data.shape)
print(data.head())

math = data['Math'].values
read = data['Reading'].values
write = data['Writing'].values

m = len(math)
x0 = np.ones(m)
X = np.array([x0, math, read]).T
# Initial Coefficients
B = np.array([0, 0, 0]) #Bad initialization
Y = np.array(write)
alpha = 0.0001

def cost_function(X, Y, B):
	m = len(Y)
	J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
	return J

inital_cost = cost_function(X, Y, B)
print("Initial Cost")
print(inital_cost)

def gradient_descent(X, Y, B, alpha, iterations):
	cost_history = [0] * (iterations + 1)
	m = len(Y)
	cost_history[0] = cost_function(X, Y, B)
	for iteration in range(iterations):
		# Hypothesis Values
		h = X.dot(B)
		# Difference b/w Hypothesis and Actual Y
		loss = h - Y
		# Gradient Calculation
		gradient = X.T.dot(loss) / m
		# Changing Values of B using Gradient
		B = B - alpha * gradient
		# New Cost Value
		cost = cost_function(X, Y, B)
		cost_history[iteration + 1] = cost
		
	return B, cost_history

# 100000 Iterations
newB, cost_history = gradient_descent(X, Y, B, alpha, 100000)

# New Values of B
print("New Coefficients")
print(newB)

# Final Cost of new B
print("Final Cost")
print(cost_history)

# Ploting the scores as scatter plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(math, read, write, color='#ef1234')
ax.plot(math, read, X.dot(newB),color='#ab0000')
plt.show()

print('init: ',cost_history[0])
iter = [i for i in range(100001)]
plt.scatter(iter, cost_history,  color='black')
plt.title('Cost')
plt.xlabel('Iter')
plt.ylabel('Cost')

plt.show()


# Model Evaluation - RMSE
def rmse(Y, Y_pred):
	rmse = np.sqrt(sum((Y - Y_pred) ** 2) / len(Y))
	return rmse

# Model Evaluation - R2 Score
def r2_score(Y, Y_pred):
	mean_y = np.mean(Y)
	ss_tot = sum((Y - mean_y) ** 2)
	ss_res = sum((Y - Y_pred) ** 2)
	r2 = 1 - (ss_res / ss_tot)
	return r2


Y_pred = X.dot(newB)

print("RMSE")
print(rmse(Y, Y_pred))
print("R2 Score")
print(r2_score(Y, Y_pred))

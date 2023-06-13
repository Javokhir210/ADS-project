import matplotlib.pyplot as plt
import pandas as pd
from LinearRegression import LinearRegression

Data = pd.read_csv('Salary_Data.csv')
print(Data)
Data.describe()


plt.scatter(Data.iloc[:, 0:1].values, Data.iloc[:, 1].values)
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.title('Plot of Data set')
plt.show()

train_size = int(0.7*Data.shape[0])
test_size = int(0.3*Data.shape[0])
print("Training set size : " + str(train_size))
print("Testing set size : "+str(test_size))


Data = Data.sample(frac=1)
X = Data.iloc[:, 0:1].values
y = Data.iloc[:, 1].values

from FeatureScaling import FeatureScaling
fs = FeatureScaling(X, y)
X = fs.fit_transform_X()
y = fs.fit_transform_Y()


X_train = X[0:train_size, :]
Y_train = y[0:train_size]

print(X_train.shape)
print(Y_train.shape)

X_test = X[train_size:, :]
Y_test = y[train_size:]


print(X_test.shape)
print(Y_test.shape)


plt.scatter(X_train, Y_train)
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.title('Plot of training set')
plt.show()

plt.scatter(X_test, Y_test)
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.title('Plot of testing set')
plt.show()




lr = LinearRegression(X_train, Y_train)

theta = lr.returnTheta()
print(theta)

y_pred_normal, error_percentage = lr.predictUsingNormalEquation(X_test, Y_test)
y_pred_normal = fs.inverse_transform_Y(y_pred_normal)
print(error_percentage)

y_pred_train_normal, error_percentage_train_normal = lr.predictUsingNormalEquation(X_train, Y_train)
y_pred_train_normal = fs.inverse_transform_Y(y_pred_train_normal)
print(lr.computeCostFunction())


n_iter = 1000
alpha = 0.05

theta, J_Array, theta_array = lr.performGradientDescent(n_iter, alpha)


y_pred_grad, ErrorPercentage = lr.predict(X_test, Y_test)
print(ErrorPercentage)
y_pred_grad = fs.inverse_transform_Y(y_pred_grad)

y_pred_train, error_for_train = lr.predict(X_train, Y_train)
y_pred_train = fs.inverse_transform_Y(y_pred_train)
print(error_for_train)

X_train = fs.inverse_transform_X(X_train)
Y_train = fs.inverse_transform_Y(Y_train)
X_test = fs.inverse_transform_X(X_test)
Y_test = fs.inverse_transform_Y(Y_test)

plt.scatter(X_train, Y_train)
plt.plot(X_train, y_pred_train, 'r')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.title('Training set prediction using Gradient Descent')
plt.show()


plt.scatter(X_train, Y_train)
plt.plot(X_train, y_pred_train_normal, 'r')
plt.xlabel('X_axis')
plt.ylabel('Y_axis')
plt.title('Training set prediction using Normal Equation')
plt.show()


plt.scatter(X_test, Y_test)
plt.plot(X_test, y_pred_grad, 'r')
plt.xlabel('X_axis')
plt.ylabel('Y_axis')
plt.title('Test set prediction using Gradient Descent')
plt.show()

plt.scatter(X_test, Y_test)
plt.plot(X_test, y_pred_normal, 'r')
plt.xlabel('X_axis')
plt.ylabel('Y_axis')
plt.title('Test set prediction using Normal Equation')
plt.show()

x = [i for i in range(1000)]
plt.plot(x, J_Array)
plt.xlabel('Number of iterations')
plt.ylabel('Cost function(J)')
plt.show()
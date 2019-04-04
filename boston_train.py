from boston_models import NeuralNet, LinearReg
import tensorflow as tf
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


def main():

	## Loading Dataset
	boston = load_boston()
	X, Y = boston['data'], boston['target']
	X, Y = shuffle(X, Y)
	X = pd.DataFrame(X)
	Y = pd.DataFrame(Y)

	## Splitting Dataset
	split = int(0.8*X.shape[0])
	x_train, y_train = X[:split], Y[:split]
	x_val, y_val = X[split:], Y[split:]

	## Standardize dataset
	sc_X = StandardScaler()
	x_train = sc_X.fit_transform(x_train)
	x_val = sc_X.transform(x_val)
	joblib.dump(sc_X, './Boston_models/sc_X.pkl')

	####################### Training Models ####################################

	# Linear Regression
	lm = LinearReg(learning_rate=0.0002)
	train_error_lm, val_error_lm = lm.train(x_train, y_train, x_val, y_val, epochs=160, batch_size=1)
	lm.sess.close()

	## Neural Network
	nn = NeuralNet(learning_rate=0.0002)
	train_error_nn, val_error_nn = nn.train(x_train, y_train, x_val, y_val, epochs=100, batch_size=1)
	nn.sess.close()

	## Support Vector Regressor
	param_grid = {"C": [1, 10, 20, 30], 
				"gamma": np.logspace(-2, 2, 5)}
	svr = SVR(gamma='scale')
	clf = GridSearchCV(svr, param_grid, cv=5)
	clf.fit(x_train, y_train)
	mse_train = mean_squared_error(y_train, clf.predict(x_train))
	mse_val = mean_squared_error(y_val, clf.predict(x_val))
	joblib.dump(clf, './Boston_models/model_svr.pkl')

	## Visualize losses
	print("Training Performance: ")
	print("LR: {0:0f} NN: {1:0f} SVR {2:0f}".format(train_error_lm[-1], train_error_nn[-1], mse_train))
	print("Validation Performance: ")
	print("LR: {0:0f} NN: {1:0f} SVR {2:0f}".format(val_error_lm[-1][0], val_error_nn[-1][0], mse_val))

	plt.plot(train_error_nn, color='red')
	plt.plot(val_error_nn, color='green')
	plt.plot(train_error_lm, color='blue')
	plt.plot(val_error_lm, color='black')
	plt.show()


if __name__ == '__main__':
	main()


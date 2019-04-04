from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from boston_train import *
from boston_models import NeuralNet, LinearReg

app = Flask(__name__)

@app.route('/json', methods=["POST"])
def json():
	''' This function is used for json inputs. Json inputs require Postman or equivalent.
		All 13 vriables need to be included with original variable names. 
		Output is Neural Network Prediction.
	'''
	inputs = []
	actual_prediction = None

	content = request.get_json()
	inputs.append(content['CRIM'])
	inputs.append(content['ZN'])
	inputs.append(content['INDUS'])
	inputs.append(content['CHAS'])
	inputs.append(content['NOX'])
	inputs.append(content['RM'])
	inputs.append(content['AGE'])
	inputs.append(content['DIS'])
	inputs.append(content['RAD'])
	inputs.append(content['TAX'])
	inputs.append(content['PTRATIO'])
	inputs.append(content['B'])
	inputs.append(content['LSTAT'])

	## Uncomment to Test on Training Dataset
	# boston = load_boston()
	# X, Y = boston['data'], boston['target']
	# X = pd.DataFrame(X)
	# Y = pd.DataFrame(Y) 
	# inputs = X.ix[500, :]
	# actual_prediction = Y.ix[500, 0]

	## Standardize data
	sc_X = joblib.load('./Boston_models/sc_X.pkl')
	x_test = np.array(inputs).reshape([1, 13])
	x_test_sc = sc_X.transform(x_test)

	## Load NN Model
	nn = NeuralNet(learning_rate=0.001) 
	nn.load_model()
	prediction_nn = nn.predict(x_test_sc)[0][0][0]

	## Load SVR Model
	clf = joblib.load('./Boston_models/model_svr.pkl')
	prediction_svr = clf.predict(x_test_sc)[0]

	return '''For Inputs: 
{0}

NN Prediction is: {1:0f} 
SVR Prediction is: {2} 
Real Price is: {3}'''.format(inputs, prediction_nn, prediction_svr, actual_prediction)



@app.route('/form', methods=['POST', 'GET'])
def form():
	''' Provided URL gives you options to manually input variables. 
		All inputs should be filled in.
		Output is a Neural Network prediction.
	'''
	input_dict = {}
	if request.method == 'POST':
		inputs = []
		crim = float(request.form.get('crim'))
		zn = float(request.form.get('zn'))
		indus = float(request.form.get('indus'))
		chas = float(request.form.get('chas'))
		nox = float(request.form.get('nox'))
		rm = float(request.form.get('rm'))
		age = float(request.form.get('age'))
		dis = float(request.form.get('dis'))
		rad = float(request.form.get('rad'))
		tax = float(request.form.get('tax'))
		ptratio = float(request.form.get('ptratio'))
		b = float(request.form.get('b'))
		lstat = float(request.form.get('lstat'))

		inputs.append([crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat])

		## Standardize data
		sc_X = joblib.load('./Boston_models/sc_X.pkl')
		x_test = np.array(inputs).reshape([1, 13])
		x_test_sc = sc_X.transform(x_test)

		## Load Model
		nn = NeuralNet(learning_rate=0.001) 
		nn.load_model()
		prediction_nn = nn.predict(x_test_sc)[0][0][0]

		## Load SVR Model
		clf = joblib.load('./Boston_models/model_svr.pkl')
		prediction_svr = clf.predict(x_test_sc)[0]
			

		return '''<p>For Inputs:</p> 
					<p>{0}<p/>

					<p>NN Prediction is: {1:0f}<p/> 
					<p>SVR Prediction is:{2:0f}<p/>'''.format(inputs, prediction_nn, prediction_svr)

	else:
		return '''<form method="POST">
				<p>Crime                 <input type="text" name="crim"><p/>
				<p>Residential Zone      <input type="text" name="zn"><p/>
				<p>Non-retail Business   <input type="text" name="indus"><p/>
				<p>Charles River         <input type="text" name="chas"><p/>
				<p>Nitric Oxides         <input type="text" name="nox"><p/>
				<p>Rooms Avg.            <input type="text" name="rm"><p/>
				<p>Age                   <input type="text" name="age"><p/>
				<p>Distance to Employment<input type="text" name="dis"><p/>
				<p>Distance to Highways  <input type="text" name="rad"><p/>
				<p>Tax                   <input type="text" name="tax"><p/>
				<p>Pupil-Teacher Ratio   <input type="text" name="ptratio"><p/>
				<p>Race Proportion       <input type="text" name="b"><p/>
				<p>Percent lower status  <input type="text" name="lstat"><p/>
				<input type="submit">  
				</form>'''


if __name__ == '__main__':
	app.run(debug=True, port=8888)

## Sample dict for prediction
# {
# 	"CRIM": 0.08199,
# 	"ZN": 0.00000,
# 	"INDUS": 13.92000,
# 	"CHAS": 0.00000,
# 	"NOX": 0.43700,
# 	"RM": 6.00900,
# 	"AGE": 42.30000,
# 	"DIS": 5.50270,
# 	"RAD": 4.00000,
# 	"TAX": 289.00000,
# 	"PTRATIO": 16.00000,
# 	"B": 396.90000,
# 	"LSTAT": 10.40000
# }
# ans=21.7
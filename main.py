import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import tensorflow as tf
import matplotlib.pyplot as plt
import math



RANDOM_SEED = 10
tf.set_random_seed(RANDOM_SEED)

def get_mixture_coef(output):
	'''================================================================================================================================
	function get_mixture_coef converts the predicted values to the conventional form used in mixture density networks, i.e.,
		mixing coeff (or pi) = softmax(pi)
		variance (or sigma) = sigmoid(sigma), generally exp(sigma) is used, but in this case sigma should lie between 0 to 1.
		mean (mu) = mu

	parameters:
		output(tensor) =  tensor of shape nX(3*m) where n is no of samples and m is no of mixtures. contains all the predicted values at neural network's output for all the input data samples.
	returns:
		out_pi(tensor) = tensor of shape nXm where n is no of samples and m is no of mixtures. contains mixing coefficients of all mixtures for every sample
		out_sigma(tensor) = tensor of shape nXm where n is no of samples and m is no of mixtures. contains variance of all mixtures for every sample
		out_mu(tensor) = tensor of shape nXm where n is no of samples and m is no of mixtures. contains means of all mixtures for every sample		
	================================================================================================================================'''

	out_pi = tf.placeholder(dtype=tf.float32, shape=[None,mixtures], name="mixparam")
	out_sigma = tf.placeholder(dtype=tf.float32, shape=[None,mixtures], name="mixparam")
	out_mu = tf.placeholder(dtype=tf.float32, shape=[None,mixtures], name="mixparam")
	out_pi, out_sigma, out_mu = tf.split(output,3,1) #split the network's output into 3 parts, mixing coeff, variance and mean.
	#folllowing two lines reduce max of mixing coeff from each mixing coeff.	
	max_pi = tf.reduce_max(out_pi, 1, keep_dims=True)
	out_pi = tf.subtract(out_pi, max_pi)
	#following three lines apply softmax function to the mixing coeff 
	out_pi = tf.exp(out_pi)
	normalize_pi = tf.divide(1,tf.reduce_sum(out_pi, 1, keep_dims=True))
	out_pi = tf.multiply(normalize_pi, out_pi)
	out_sigma = tf.exp(out_sigma) # variance is exp(variance)
	return out_pi, out_sigma, out_mu



def tf_normal(y, mu, sigma):
	'''================================================================================================================================
	function tf_normal calculates the probability (Gaussian) of y given mean mu and variance sigma, i.e.,

		prob= 1/sigma*sqrt(2*pi)*{exp(-0.5*((y-mu)/sigma)^2)}

	parameters:
		y(tensor) =  tensor of shape nX1 where n is no of samples. contains actual target
		mu(tensor) = tensor of shape nXm where n is no of samples and m is no of mixtures. contains means of all mixtures for every sample		
		sigma(tensor) = tensor of shape nXm where n is no of samples and m is no of mixtures. contains variance of all mixtures for every sample
	returns:
		prob(Integer) = tensor of shape nXm where n is no of samples and m is no of mixtures. contains probability (Gaussian) of y given mean, mu and variance, sigma.
		
	================================================================================================================================'''

	oneDivSqrtTwoPI = 1 / math.sqrt(2*math.pi) # normalisation factor for gaussian.
	result = tf.subtract(y, mu) #mean subtraction
	result = tf.multiply(result, tf.divide(1, sigma)) #divide by variance
	result = -tf.square(result) / 2
	prob = tf.multiply(tf.exp(result), tf.divide(1, sigma)) * oneDivSqrtTwoPI #normalizing using normalisation factor for gaussian.
	return prob

def get_lossfunc(out_pi, out_sigma, out_mu, y):
	'''================================================================================================================================
	function get_lossfunc returns the mean negetive log liklihood error from the predicted distribution and actual target y.

	parameters:
		out_pi(tensor) = tensor of shape nXm where n is no of samples and m is no of mixtures. contains mixing coefficients of all mixtures for every sample
		out_sigma(tensor) = tensor of shape nXm where n is no of samples and m is no of mixtures. contains variance of all mixtures for every sample
		out_mu(tensor) = tensor of shape nXm where n is no of samples and m is no of mixtures. contains means of all mixtures for every sample		
		y(tensor) =  tensor of shape nX1 where n is no of samples. contains actual target
	returns:
		result(tensor) = mean negetive log liklihood error
		
	================================================================================================================================'''

	result = tf_normal(y, out_mu, out_sigma) #returns Gaussian probability of y given mu and sigma.
	result = tf.multiply(result, out_pi) #multiply each mixture density with corresponding mixing coefficients.
	result = tf.reduce_sum(result, 1, keep_dims=True) #weighted sum
	result = -tf.log(result) #negetive log liklihood of the conditional probability (weighted sum of individual mixture conditional probability)
	error = tf.reduce_mean(result) #mean negetive log liklihood
	return error
		
def get_pi_idx(x, pdf, m):
	'''================================================================================================================================
	function get_pi_idx returns the mean (for each input) of the sampled mixture component

	parameters:
		x(Float) = a random nuumber
		pdf(array) = array of mixing coefficients (as probabilities) of mixture components.
		m(array) = array of means of mixture components.	
	returns:
		i(Integer) = index of ith distribution which is sampled.
		
	================================================================================================================================'''

	N = pdf.size
	accumulate = 0
	for i in range(0, N): #for all mixtures
		accumulate += pdf[i] #add the mixing coefficients (probabilities)
		if (accumulate >= x): #till accumulate>=x
			if((m[i] <= 1e-2 or m[i] == float(1)) and i > 0):					
				return i-1
			elif((m[i] <= 1e-2 or m[i] == float(1))and i < N-1):
				return i-1
			else:
				return i
	print 'error with sampling ensemble'
	return -1



def sample(out_pi, out_mu, out_sigma,NTEST):
	'''================================================================================================================================
	function sample uses "sampling" method to return the mean (for each input) of the sampled mixture component

	parameters:
		out_pi(array) = array of dimension nXm where n is no of samples and m is no of mixtures. contains mixing coefficients of all mixtures for every sample
		out_mu(array) = array of dimension nXm where n is no of samples and m is no of mixtures. contains means of all mixtures for every sample
		out_sigma(array) = array of dimension nXm where n is no of samples and m is no of mixtures. contains variance of all mixtures for every sample
		NTEST(Integer) = no. of test samples
	returns:
		result(array) = (nX1 dimension) mean of the sampled mixture component dist., for each input
		
	================================================================================================================================'''

	result = np.random.rand(NTEST, 1) #result array initialized to random values
	for j in range(0, 1):
		for i in range(0, NTEST):
			params = np.array(np.transpose([out_pi[i], out_mu[i], out_sigma[i]])) 
			rand = np.random.rand() #generate a random no between 0 and 1 uniformly.
 			idx = get_pi_idx(rand, params[:,0], params[:,1]) #returns index (0 to m-1) of the sampled mixture component dist.
			mu = params[idx][1]
			result[i, j] = mu
	return result



def readData(inpath, wells, skipr):
	'''================================================================================================================================
	function readData reads well data from .csv files and concatenates them.

	parameters:
		inpath(String) = path for input data.
		skipr(Integer) = no. of rows to skip.
	returns:
		train(array) = training data
		test(array) = test data (last well in wells list).
		
	================================================================================================================================'''
	ds = []
	for i in range(wells):
		a = pd.read_csv(inpath + "well" + i + ".csv", header = None, skiprows = 1)
		ds.append(a)
	df = ds[0]
	for i in range(1, len(ds) - 1):
		df = pd.concat([df, ds[i]])
	train = df.values.astype('float64')
	test = ds[-1].values.astype('float64')
	return train, test




def mdn(Input, hidden, mixtures, learning_rate, epoch, inpath, wells, skipr,  skipc):
	'''=================================================================================================================================
	function mdn contains an implementation of Mixture Density network. 

	parameters:
		Input(Integer) = dimension of inputs
		hidden(Integer) = no. of hidden neurons
		mixtures(Integer) = no. of mixtures
		learning_rate(Float) = specify the learning_rate for training
		epoch(Integer) = no. of epochs to train
		inpath(String) = path for input data		
		skipr(Integer) = no. of rows to skip (Default=1)
		skipc(Integer) = no. of columns to skip (Default=0)
	output:
		RMSE and Plots of predicted and actual porosity for test set.
	=================================================================================================================================='''


	output=mixtures*3

	#read data
	train, test=readData(inpath, wells, skipr)
	
	#define placeholders
	X = tf.placeholder(tf.float32, [None, Input])
	Y = tf.placeholder(tf.float32, [None, 1])	
			
	#intialize parameters
	W1 = tf.get_variable('W1', initializer=tf.random_normal([Input, hidden], mean = 0.0, stddev = 1))
	W3 = tf.get_variable('W3', initializer=tf.random_normal([hidden, output], mean = 0.0, stddev = 1))
	B1 = tf.get_variable('B1', initializer=tf.ones((hidden)))
	B3 = tf.get_variable('B3', initializer=tf.ones((output)))
	
	#hidden layer 1
	A1 = tf.nn.xw_plus_b(X, W1, B1)
	H1 = tf.nn.sigmoid(A1)
		
	#output layer
	A3 = tf.nn.xw_plus_b(H1, W3, B3)
	H3 = tf.nn.sigmoid(A3)

	#mixing coefficient,  	
	out_pi, out_sigma, out_mu = get_mixture_coef(H3)
	
	#loss and optimization
	loss = get_lossfunc(out_pi, out_sigma, out_mu, Y)
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
		
	#creating session
	with tf.Session() as sess:
		#training 
		sess.run(tf.global_variables_initializer())
		train_x = train[:, skipc:skipc+Input] #train input
		train_y = train[:, train.shape[1]-1:train.shape[1]] #train target
		test_x = test[:, skipc:skipc+Input] #test input
		test_y = test[:, test.shape[1]-1:test.shape[1]] #test target

		print("training loss:\n\t")
		lossfunc = np.zeros(epoch) #array of training loss
		for i in range(epoch):
			_, lossfunc[i] = sess.run([optimizer, loss], {X:(train_x), Y:(train_y)}) #train and return loss
			print ("epoch", i, ":", lossfunc[i])


		
		l, mix_coef, var, mean = sess.run([loss, out_pi, out_sigma, out_mu], {X:(test_x), Y:(test_y)}) #test and return mixture dist parameters.
		pred = np.zeros(mean.shape[0])
		for j in range(mean.shape[0]):
			pred[j] = mean[j][np.argmax(mix_coef[j] / var[i])] #predict the mean of that mixture component whose mixing_coeff/variance ratio is highest.
		residual = np.absolute(np.subtract(test_y, pred)) #calculate error residuals
		mse = np.sqrt(np.mean(np.square(residual)))  # calculate mean squared error
		print "MSE of max_coeff method: - ", mse

		pred2 = sample(mix_coef, mean, var, test_x.shape[0])
		residual = np.absolute(np.subtract(test_y, pred2))
		mse1 = np.sqrt(np.mean(np.square(residual)))
		print "MSE of sampling method: - ",mse1

		resultFile = open("results.txt", "w")
		resultFile.write("test error using high-coeff method : "+str(mse))
		resultFile.write("\n test error using sampling method : "+str(mse1))
	


if __name__ == "__main__":
    mixtures = 5
    epoch = 100
    learning_rate = 0.01
    Input = 14
    hidden = 150
    inpath = "Data/"
    wells = 7
    mdn(Input, hidden, mixtures, learning_rate, epoch, inpath, wells, skipr=1, skipc=0)







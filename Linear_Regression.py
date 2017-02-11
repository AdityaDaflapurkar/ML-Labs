import numpy as np

class LinearReg:
	def __init__(self):
		#randomly initialize weights here
		self.theta=np.random.uniform(-100,100,[1,3])

	def gradient(self,X, Y):
		#returns gradient, MSE
		m=len(X)		
		h=self.predict(X)
		MSE=np.sum((h-Y)**2)/m
		grad=(2*np.dot(h-Y,X))/m
		return grad,MSE

	def predict(self,X):
		#returns predictions for inputs X
		return np.dot(self.theta,np.transpose(X))

	def train(self,X,Y, numberOfIterations, lr):
		#gradient descent code here
		for i in xrange(numberOfIterations):
			gradient_value,MSE=self.gradient(X,Y)
			self.theta=self.theta-gradient_value*lr
			#print "Mean square error : ",MSE,"	iteration : ",i+1
		

if __name__ == '__main__':
	#intialize data and perform train, validation, test splits
	#call train and predict
	c=np.array([2,5,3])
	m=100		
	X=np.random.uniform(1,100,[m,2])
	X=np.insert(X,0,1,axis=1)
	error=np.random.uniform(-1,1,[1,100])
	Y=np.dot(c,np.transpose(X))+error	  ### y=c0+5*x1+3*x2+error

	tempY=np.transpose(Y)
	tempErr=np.transpose(error)	
	
	train_X=X[0:int(0.6*m)]
	error_train=np.transpose(tempErr[0:int(0.6*m)])
	train_Y=np.transpose(tempY[0:int(0.6*m)])
	
	test_X=X[int(0.6*m):int(0.8*m)]
	error_test=np.transpose(tempErr[int(0.6*m):int(0.8*m)])
	test_Y=np.transpose(tempY[int(0.6*m):int(0.8*m)])
	
	validate_X=X[int(0.8*m):]
	error_validate=np.transpose(tempErr[int(0.8*m):])
	validate_Y=np.transpose(tempY[int(0.8*m):])
	

	lr=0.0001
	model=LinearReg()
	print "Training model..."	
	model.train(train_X,train_Y,300000,lr)
	
	print "Actual weights : ",c
	print "Predicted weights : ",model.theta[0]
	
	print "Performance with test set:"
	test_predicted=model.predict(test_X)
	print "Actual value","	","Predicted value"
	for i in xrange(int(0.2*m)):
		print test_Y[0][i],"	",test_predicted[0][i]

	print "Performance with validation set:"
	validate_predicted=model.predict(validate_X)
	print "Actual value","	","Predicted value"
	for i in xrange(int(0.2*m)):
		print validate_Y[0][i],"	",validate_predicted[0][i]



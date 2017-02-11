import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

class LogisticReg:
	def __init__(self):
		self.theta=np.random.uniform(-1,1,[1,3])
		self.Lambda=10
		self.error=[]
		
	def normalize(self,X):
		avg=np.mean(X[:,0:len(X[0])-1],axis=0)
		maxi=np.amax(X[:,0:len(X[0])-1])
		mini=np.amin(X[:,0:len(X[0])-1])
		X[:,0:len(X[0])-1]=(X[:,0:len(X[0])-1]-avg)/(maxi-mini)
		return X

	def testSplit(self,X):
		train=X[0:int(0.6*len(X))]
		validate=X[int(0.6*len(X)):int(0.8*len(X))]
		test=X[int(0.8*len(X)):]
		return train,validate,test
	
	def sigmoid(self,X):
		return 1/(1+np.exp(-X))
	
	def gradient(self,X,Y):
		m=len(X)
		h=self.predict(X)
		mask = np.ones(len(self.theta));
		mask[0] = 0
		cost=-(np.sum(Y*np.log(h)+(1-Y)*np.log(1-h))/m)+(self.Lambda*(np.sum(self.theta**2))*mask/(2*m))
		grad=np.dot(h-Y,X)/m+(self.Lambda*self.theta*mask)/m		
		return grad,cost

		
	def train(self,X,Y,iterations,lr):
		for i in xrange(iterations):
			gradient_value,cost=self.gradient(X,Y)
			self.theta=self.theta-gradient_value*lr
			#print np.shape(self.theta),"t"
			#print "Cost : ",cost,"gradient : ",gradient_value," iteration : ",i+1
			self.error.append(cost)
		
	def predict(self,X):
		h=self.sigmoid(np.dot(self.theta,X.T))
		return h

	def step(self,x):
		return np.piecewise(x,[x>=0.5,x<0.5],[1,0])
	
if __name__ == '__main__':
	df=pd.read_csv('dataset.csv',header=None)
	df=shuffle(df)
	X=df.as_matrix(columns=None)
	model=LogisticReg()
	X=model.normalize(X)
	X=np.insert(X,0,1,axis=1)
	trainSet,validationSet,testSet=model.testSplit(X)
	X=trainSet[:,0:len(trainSet[0])-1]
	Y=trainSet[:,-1]
	model.train(X,Y,1000,0.1)
	X=validationSet[:,0:len(testSet[0])-1]
	Y=validationSet[:,-1]
	p=model.predict(X)
	accv=0
	print "Performance on Validation set :"
	print "Expected class Predicted class"
	vl=len(Y)	
	for i in xrange(vl):
		ex=int(Y[i])
		pred=int(model.step(p[0][i]))
		print ex,"		",pred
		if ex==pred:
			accv=accv+1
	X=testSet[:,0:len(testSet[0])-1]
	Y=testSet[:,-1]
	p=model.predict(X)
	acct=0
	print "Performance on Test set :"
	print "Expected class Predicted class"	
	tl=len(Y)
	for i in xrange(tl):
		ex=int(Y[i])
		pred=int(model.step(p[0][i]))
		print ex,"		",pred
		if ex==pred:
			acct=acct+1 
	print "Accuracy on validation set = ",(accv*100)/vl," percent"	
	print "Accuracy on test set = ",(acct*100)/tl," percent"
	

	#fig = plt.figure()
	plt.plot(range(1000),model.error)
	plt.xlabel('No. of Iterations')
	plt.ylabel('Cost Function')
	plt.title('Gradient Descent Cost Function Plot')
	plt.show()

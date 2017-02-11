import numpy as np
import math
import operator as op
import pandas as pd
from sklearn.utils import shuffle

class knn:

	def __init__(self,K):
		self.training_set=list()
		self.k=K

	def train(self,x):
		self.training_set.extend(X)

	def l1_distance(self,x1,x2):
		d=0
		for i in xrange(len(x1)-1):
			d=d+abs(x1[i]-x2[i])
		return d

	def l2_distance(self,x1,x2):
		d=0
		for i in xrange(len(x1)-1):
			d=d+(x1[i]-x2[i])**2
		return math.sqrt(d)
	
	def kNearestNeighbors(self,x):
		dist=[]
		for i in xrange(len(self.training_set)):
		       	#d=self.l1_distance(x,self.training_set[i])
			d=self.l2_distance(x,self.training_set[i])
			dist.append((self.training_set[i],d))
		dist.sort(key=op.itemgetter(1))
		neighbors = []
		for i in range(self.k):
			neighbors.append(dist[i][0])
		return neighbors

	def predict(self,x):
		votes={"setosa":0,"versicolor":0,"virginica":0}
		neighbors=self.kNearestNeighbors(x)		
		for i in xrange(len(neighbors)):
			votes[neighbors[i][-1]]+=1
		sortedVotes = sorted(votes.iteritems(), key=op.itemgetter(1), reverse=True)
		return sortedVotes[0][0]
		
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
		
if __name__ == '__main__':
	df=pd.read_csv('IRIS.csv',header=None)
	df=shuffle(df)
	X=df.as_matrix(columns=None)
	model=knn(3)
	X=model.normalize(X)
	trainSet,validationSet,testSet=model.testSplit(X)
	for i in xrange(len(trainSet)):
		model.train(trainSet[i])
	
	print "Performance on Validation set :"
	print "================================="
	for i in xrange(len(validationSet)):
		print "Row :",i+1
		print "Actual class : ",validationSet[i][-1]
		print "Predicted class : ",model.predict(validationSet[i])
		print ""

	print "Performance on Test set :"
	print "================================="
	for i in xrange(len(testSet)):
		print "Row :",i+1
		print "Actual class : ",testSet[i][-1]
		print "Predicted class : ",model.predict(testSet[i])
		print ""

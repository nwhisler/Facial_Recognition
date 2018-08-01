from process_data import get_data
from sklearn.utils import shuffle
from scipy.stats import multivariate_normal as mvn
import numpy as np, matplotlib.pyplot as plt

class Facial_Rec(object):

	def fit(self,X,Y,smoothing=10e-3):

		self.gaussian = {}
		self.prior = {}
		self.labels = set(Y)
		for i in self.labels:
			current_X = X[Y == i]
			self.gaussian[i] = {
				'mean':current_X.mean(axis=0),
				'cov':np.cov(current_X.T) + np.eye(len(current_X.T))*smoothing
			}
			self.prior[i] = float(len(Y[Y==i]))/len(Y)

	def project(self,X):

		N = len(X)
		D = len(self.gaussian)
		P = np.zeros((N,D))
		for i in self.labels:
			mean = self.gaussian[i]['mean']
			cov = self.gaussian[i]['cov']
			P[:,i] = mvn.logpdf(X,mean=mean,cov=cov) + np.log(self.prior[i])
		return np.argmax(P,axis=1)
	def score(self,X,Y):

		P = self.project(X)
		return np.mean(Y == P)


if __name__ == '__main__':

	X,Y = get_data()
	X,Y = shuffle(X,Y)
	N = len(Y)//2
	Xtrain = X[:N]
	Ytrain = Y[:N]
	Xtest = X[N:]
	Ytest = Y[N:]
	model = Facial_Rec()
	model.fit(Xtrain,Ytrain)
	print('Train accuracy: ',model.score(Xtrain,Ytrain))
	print('Test accuracy: ',model.score(Xtest,Ytest))
	print()
	alphabet = np.array([chr(i) for i in range(65,91)])
	idx = [22,7,8,18,11,4,17]
	delim = ''
	print(delim.join(alphabet[idx]))
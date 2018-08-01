from sklearn.utils import shuffle
import pandas as pd, matplotlib.pyplot as plt, numpy as np

def get_data():

	df = pd.read_csv('faces.csv')
	Y = df['emotion'].as_matrix()
	pixels = [i.split(' ') for i in list(df['pixels'])]
	X = np.array(pixels,dtype=np.float64)
	labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
	for i in range(len(Y)):
	 	plt.imshow(X[i].reshape(48,48),cmap='gray')
	 	plt.title('%s' %labels[Y[i]])
	 	plt.show()
	 	if input('Exit: Y: ') == 'Y':
	 		break
	additional_feature1 = np.zeros(len(Y))
	additional_feature2 = np.zeros(len(Y))
	additional_feature3 = np.zeros(len(Y))
	additional_feature4 = np.zeros(len(Y))
	additional_feature5 = np.zeros(len(Y))
	additional_feature6 = np.zeros(len(Y))
	additional_feature7 = np.zeros(len(Y))
	additional_feature8 = np.zeros(len(Y))
	for i in set(Y):
		idx = np.nonzero(Y == i)[0]
		avg_pixel = X[idx].mean()
		std_pixel = X[idx].std()
		var_pixel = X[idx].var()
		feature1 = avg_pixel * std_pixel
		feature2 = avg_pixel * var_pixel
		feature3 = std_pixel * var_pixel
		feature4 = avg_pixel * std_pixel * var_pixel
		feature5 = feature1 * feature2 * feature3 * feature4
		additional_feature1[idx] = avg_pixel
		additional_feature2[idx] = std_pixel
		additional_feature3[idx] = var_pixel
		additional_feature4[idx] = feature1
		additional_feature5[idx] = feature2
		additional_feature6[idx] = feature3
		additional_feature7[idx] = feature4
		additional_feature8[idx] = feature5
	additional_feature1 = np.array([additional_feature1]).T
	additional_feature2 = np.array([additional_feature2]).T
	additional_feature3 = np.array([additional_feature3]).T
	additional_feature4 = np.array([additional_feature4]).T
	additional_feature5 = np.array([additional_feature5]).T
	additional_feature6 = np.array([additional_feature6]).T
	additional_feature7 = np.array([additional_feature7]).T
	additional_feature8 = np.array([additional_feature8]).T
	X = np.concatenate((X,additional_feature1,additional_feature2,additional_feature3,additional_feature4,
		additional_feature5,additional_feature6,additional_feature7,additional_feature8),axis=1)
	X = (X - X.mean())/X.std()
	return X,Y
import pandas as pd
import os
import numpy as np
import math
from sklearn import datasets # to import the iris dataset
import pickle

class plus1_sigma:

	def __init__(self, sigma, w_old, lambda_value, maxiter, tol):
		self.sigma = sigma
		self.w_old = w_old
		self.lambda_value = lambda_value
		self.maxiter = maxiter
		self.tol = tol
		self.Phi =  self.sigma - np.diag(np.repeat(1, self.sigma.shape[1]))
		self.w_new = w_old


	# norm function 
	def norm(self, x):
		result = math.sqrt(np.sum(x**2))
		return result


	def H(self):

		Phi_w = np.matmul(self.Phi, self.w_old)
	
		fraction = Phi_w / self.norm(Phi_w)

		to_check = (Phi_w)**2 / self.norm(Phi_w)

		threshold = to_check <= (self.lambda_value / 2)

		fraction[threshold] = 0

		return fraction


	def plus1_sigma (self):

		obj_orig_old = np.transpose(self.w_old) @ self.sigma @ self.w_old - self.lambda_value * np.sum(self.w_old != 0)

		# pre-defining the objects to be filled with the iterates
		obj_orig_hist = np.repeat(float(-1), self.maxiter)

		W_hist = np.ones((self.sigma.shape[1], self.maxiter)) * -1

		obj_orig_hist[0] = obj_orig_old

		W_hist[: , 0] = self.w_old

		diff_norm_hist = np.repeat(float(-1), self.maxiter)

		iii = 0

		while True:

			self.w_new = self.H()

			# if zero vector is resulted, stop the iteration
			if np.sum(self.w_new == 0) != self.w_new.shape[0]:
				self.w_new = self.w_new / self.norm(self.w_new)
			else:
				print ("zero vector resulted")
				break

			# if max iteration is reached, stop the iteration
			if iii > 2 and iii > self.maxiter:
			  	print("max iteration reached")
			  	break

			# comparing the norm between w_old and w_new to see if stop:
			if iii > 2 and self.norm(self.w_old - self.w_new) < tol:
				print ("w norm condition suffices")

				iii = iii + 1

				obj_orig_new = np.transpose(self.w_new) @ self.sigma @ self.w_new - lambda_value * np.sum(self.w_new != 0)

				obj_orig_hist[iii] = obj_orig_new

				diff_norm_hist[iii] = self.norm(self.w_old - self.w_new)

				W_hist[:, iii] = self.w_new

				break

			iii = iii + 1

			obj_orig_new = np.transpose(self.w_new) @ self.sigma @ self.w_new - lambda_value * np.sum(self.w_new != 0)

			obj_orig_hist[iii] = obj_orig_new

			diff_norm_hist[iii] = self.norm(self.w_old - self.w_new)

			W_hist[:, iii] = self.w_new

			if obj_orig_new - obj_orig_old < 0:
				print ("original objective decrease at " + str(iii))


			# updating the w iterate
			self.w_old = self.w_new

			# updating the objectives
			obj_orig_old = obj_orig_new

		# reporting only the relevant results in the history objects
		obj_orig_hist = obj_orig_hist[:(iii+1)]
		diff_norm_hist = diff_norm_hist[:(iii+1)]
		W_hist = W_hist[:, :(iii + 1)]

		results = dict();
		results["w_new"] = self.w_new    	
		results["W_hist"] = W_hist
		results["obj_hist"] = obj_orig_hist
		results["obj"] = obj_orig_new
		results["diff_norm"] = diff_norm_hist
		results["i"] = iii

		return results


# defining another class that takes w_new result as input and lets out the component score as output
# this class will be saved as pickle object
class plus1_predict:

	def __init__(self, w_new):
		self.w_new = w_new

	def predict(self, newX):
		tscore = np.matmul(newX, self.w_new)
		return tscore


# iris data load
iris = datasets.load_iris()

X = iris.data

# initializing the function
w_old = np.array([0.5,-0.5,1,9])
lambda_value = 1000
Sigma = np.matmul(np.transpose(X), X)
maxiter = 1000
tol = 1e-12

# training the data
model = plus1_sigma(Sigma, w_old, lambda_value, maxiter, tol)

fitted = model.plus1_sigma()

print(fitted["w_new"])

# predicting class
plus1_predicting = plus1_predict(fitted["w_new"])


#create a pickle file
picklefile = open('plus1_pickled.pkl', 'wb')

#pickle the class and write to the file
pickle.dump(plus1_predicting, picklefile)

#close the file
picklefile.close()



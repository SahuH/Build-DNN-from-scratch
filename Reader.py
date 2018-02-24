#!/usr/bin/env python

from mnist import MNIST
import random
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import time
import os


class Reader(object):

	def __init__(self):
		self.train_images = None 
		self.test_images = None 
		self.train_labels = None
		self.test_labels = None 
		self.num_train_example = None
		self.num_test_example  = None
		self.train_label_vector = None
		self.test_label_vector = None
		self.num_features = None	
		self.class_var = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
		self.num_labels = len(self.class_var) 
		self.output_dim = len(self.class_var)

	def read(self):
		mndata = MNIST()

		print "reading data.."
		training_images, training_labels = mndata.load_training()
		print "training ubyte files read..."
		testing_images, testing_labels = mndata.load_testing()
		print "data read successfully from ubyte files"
		train_images = np.array(training_images, dtype=np.float64)
		print "training images numpy array created..."
		train_labels = np.array(training_labels, dtype=np.float64)
		print "train_labels numpy array created..."
		test_images = np.array(testing_images, dtype=np.float64)
		print "test_images numpy array created..."
		test_labels = np.array(testing_labels, dtype=np.float64)
		print "test_labels numpy array created..."

		#filter out training variables 
		self.train_labels = np.array(filter(lambda x: x in self.class_var, train_labels))
		print "shape of train_labels is %s" %str(self.train_labels.shape)
		self.num_train_example = self.train_labels.shape[0]

		self.test_labels =  np.array(filter(lambda x: x in self.class_var, test_labels))
		self.num_test_example = self.test_labels.shape[0]

		train_indices = [index for index,value in enumerate(train_labels) if value in self.class_var]
		train_images_temp = train_images[train_indices,:]
		self.train_images = train_images_temp/256.0

		test_indices = [index for index,value in enumerate(test_labels) if value in self.class_var]
		test_images_temp = test_images[test_indices,:]
		self.test_images = test_images_temp/256.0
		self.num_features = self.train_images.shape[1]	
		print "calling vectorise output now..."
		self.vectorise_output()
		#self.find_balance()

	def find_balance(self):
		l = []
		for i in range(9):
			c=0
			for label in self.test_labels:
				if label==i:
					c+=1
			l.append(c)
		print l

	def vectorise_output(self):
		self.train_label_vector = np.zeros((len(self.train_labels), self.output_dim), dtype=np.float64)
		self.test_label_vector = np.zeros((len(self.test_labels), self.output_dim), dtype=np.float64)
		label_indices_list_train = []
		label_indices_list_test = []

		for i in range(0,self.num_labels):
			label_indices_list_train.append([index for index,value in enumerate(self.train_labels) if value in [self.class_var[i]]])
		for j in range(0,self.num_labels):
			label_indices_list_test.append([index for index,value in enumerate(self.test_labels) if value in [self.class_var[j]]])
		
		
		for k in range(0,self.num_labels):
			for t in label_indices_list_train[k]:
				temp = np.zeros(self.num_labels)
				temp[k] = 1.0
				self.train_label_vector[t, :] = temp
				
		for l in range(0,self.num_labels):
			for u in label_indices_list_test[l]:
				temp = np.zeros(self.num_labels)
				temp[l] = 1.0
				self.test_label_vector[u, :] = temp	

		print "output vectors created for training..."
		
if '__name__' == '__main__':
	obj = Reader()
	obj.read()




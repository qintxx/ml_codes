#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

__author__="Qinxue"

import numpy as np

def multidimensionalgaussian(x,u,sigma,dimension):
	print 'print x...'
	print x
	print 'print u...'
	print u
	print 'print sigma...'
	print sigma
	print 'print dimension'
	print dimension
	return 1.0/(np.power(2.0*np.pi,dimension/2.0)\
	*np.sqrt(np.linalg.det(sigma)))*np.exp(-1.0/2.0*\
		np.dot(np.dot(x-u,np.linalg.inv(sigma)),x-u))

def norm(x):
	return np.sqrt(np.dot(x,x))

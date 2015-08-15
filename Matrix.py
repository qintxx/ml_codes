#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

__author__="Qinxue"

import numpy as np

def multipy(A,B):
	ma,na=A.shape
	mb,nb=B.shape
	if na != mb:
		raise Exception('dimension does not match')
	C=np.zeros(ma*nb).reshape(ma,nb)
	for i in range(ma):
		for j in range(nb):
			C[i][j]=np.dot(A[i,:],B[:,j])
	return C

def vTvmultipy(x,y):
	m=len(x)
	n=len(y)
	C=np.zeros(m*n).reshape(m,n)
	for i in range(m):
		for j in range(n):
			C[i][j]=x[i]*y[j]
	return C
